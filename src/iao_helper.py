'''
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters
    
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

import qcdmet_paths
from pyscf import gto
import numpy as np
import scipy

def construct_p_list( mol, pmol ):

    Norbs       = mol.nao_nr()
    p_list      = np.zeros( [ Norbs ], dtype=int )
    counter_mol = 0
    
    for item in mol.spheric_labels():
        for pitem in pmol.spheric_labels():
            if (( pitem[0] == item[0] ) and ( pitem[1] == item[1] ) and ( pitem[2] == item[2] ) and ( pitem[3] == item[3] )):
                p_list[ counter_mol ] = 1
        counter_mol += 1
    
    assert( counter_mol == Norbs )
    assert( np.sum( p_list ) == pmol.nao_nr() )
    return p_list

def orthogonalize_iao( coeff, ovlp ):

    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C, third equation
    eigs, vecs = scipy.linalg.eigh( np.dot( coeff.T, np.dot( ovlp, coeff ) ) )
    coeff      = np.dot( coeff, np.dot( np.dot( vecs, np.diag( np.power( eigs, -0.5 ) ) ), vecs.T ) )
    return coeff
    
def resort_orbitals( mol, ao2loc ):

    # Sort the orbitals according to the atom list
    Norbs  = mol.nao_nr()
    coords = np.zeros( [ Norbs, 3 ], dtype=float )
    rvec   = mol.intor( 'cint1e_r_sph', 3 )
    for cart in range(3):
        coords[ :, cart ] = np.diag( np.dot( np.dot( ao2loc.T, rvec[cart] ) , ao2loc ) )
    atomid = np.zeros( [ Norbs ], dtype=int )
    for orb in range( Norbs ):
        min_id = 0
        min_distance = np.linalg.norm( coords[ orb, : ] - mol.atom_coord( 0 ) )
        for atom in range( 1, mol.natm ):
            current_distance = np.linalg.norm( coords[ orb, : ] - mol.atom_coord( atom ) )
            if ( current_distance < min_distance ):
                min_distance = current_distance
                min_id = atom
        atomid[ orb ] = min_id
    resort = []
    for atom in range( 0, mol.natm ):
        for orb in range( Norbs ):
            if ( atomid[ orb ] == atom ):
                resort.append( orb )
    resort = np.array( resort )
    ao2loc = ao2loc[ :, resort ]
    return ao2loc
    
def construct_iao( mol, mf ):

    Norbs = mol.nao_nr()

    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C
    ao2occ = mf.mo_coeff[ :, mf.mo_occ > 0.5 ]
    pmol   = mol.copy()
    pmol.build( False, False, basis='minao' )
    S21    = gto.mole.intor_cross( 'cint1e_ovlp_sph', pmol, mol )
    S1     = mol.intor('cint1e_ovlp_sph')
    S2     = pmol.intor('cint1e_ovlp_sph')
    X      = np.linalg.solve( S2, np.dot( S21, ao2occ ) )
    P12    = np.linalg.solve( S1, S21.T )
    Cp     = np.dot( P12, X )
    Cp     = orthogonalize_iao( Cp, S1 )
    DM1    = np.dot( ao2occ, ao2occ.T )
    DM2    = np.dot( Cp, Cp.T )
    A      = 2 * np.dot( DM1, np.dot( S1, np.dot( DM2, S21.T ) ) ) + P12 - np.dot( DM1 + DM2, S21.T )
    ao2iao = orthogonalize_iao( A, S1 )
    return ( ao2iao , S1, pmol )

def localize_iao( mol, mf ):

    Norbs = mol.nao_nr()
    ao2iao, S1, pmol = construct_iao( mol, mf )
    num_iao = ao2iao.shape[ 1 ]

    # Determine the complement of the IAO space
    DM_iao     = np.dot( ao2iao, ao2iao.T )
    mx         = np.dot( S1, np.dot( DM_iao, S1 ) )
    eigs, vecs = scipy.linalg.eigh( a=mx, b=S1 ) # Small to large in scipy
    ao2com     = vecs[ :, : Norbs - num_iao ]

    # Redo the IAO contruction for the complement space
    p_list = construct_p_list( mol, pmol ) # return array of length Norbs; 1 if similar bf in pmol; 0 otherwise
    S31    = S1[  p_list == 0 , : ]
    S3     = S31[ : , p_list == 0 ]
    X      = np.linalg.solve( S3, np.dot( S31, ao2com ) )
    P13    = np.linalg.solve( S1, S31.T )
    Cp     = np.dot( P13, X )
    Cp     = orthogonalize_iao( Cp, S1 )
    DM1    = np.dot( ao2com, ao2com.T )
    DM3    = np.dot( Cp, Cp.T )
    A      = 2 * np.dot( DM1, np.dot( S1, np.dot( DM3, S31.T ) ) ) + P13 - np.dot( DM1 + DM3, S31.T )
    ao2com = orthogonalize_iao( A, S1 )
    ao2loc = np.hstack( ( ao2iao, ao2com ) )
    ao2loc = resort_orbitals( mol, ao2loc )
    ao2loc = orthogonalize_iao( ao2loc, S1 )
    
    # Quick check
    should_be_1 = np.dot( np.dot( ao2loc.T, S1 ), ao2loc )
    print("QC-DMET :: iao_helper :: num_orb pmol =", pmol.nao_nr())
    print("QC-DMET :: iao_helper :: num_orb mol  =", mol.nao_nr())
    print("QC-DMET :: iao_helper :: norm( I - C_full.T * S * C_full ) =", np.linalg.norm( should_be_1 - np.eye( should_be_1.shape[0] ) ))
    
    return ao2loc
    
    
