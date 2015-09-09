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

import sys
sys.path.append('/home/seba') #Folder in which PySCF is installed
from pyscf import gto, scf, tools
import pyscf.lib.parameters as param
from pyscf.tools import localizer
import numpy as np
import scipy

def minao_basis( symb, minao ):

    # Copied from pyscf/future/lo/iao.py:
    basis_add = gto.basis.load(minao, symb)
    basis_new = []
    for l in range(4):
        nuc = gto.mole._charge(symb)
        ne = param.ELEMENTS[nuc][2][l]
        nd = (l * 2 + 1) * 2
        nshell = int(np.ceil(float(ne)/nd))
        if nshell > 0:
            basis_new.append([l] + [b[:nshell+1] for b in basis_add[l][1:]])
    return basis_new

def minao_mol( mol, minao='minao' ):

    # Copied from pyscf/future/lo/iao.py:
    atmlst = set([gto.mole._rm_digit(gto.mole._symbol(k)) for k in mol.basis.keys()])
    basis = {}
    for symb in atmlst:
        basis[symb] = minao_basis(symb, minao)

    pmol = gto.Mole()
    pmol._atm, pmol._bas, pmol._env = pmol.make_env(mol.atom, basis, [])
    pmol.natm = len(pmol._atm)
    pmol.nbas = len(pmol._bas)
    return pmol

def orthogonalize_iao( coeff, ovlp ):

    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C, third equation
    eigs, vecs = np.linalg.eigh( np.dot( coeff.T, np.dot( ovlp, coeff ) ) )
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

def localize_iao( mol, mf ):

    Norbs = mol.nao_nr()

    # Knizia, JCTC 9, 4834-4843, 2013 -- appendix C
    ao2occ = mf.mo_coeff[ :, mf.mo_occ > 0.5 ]
    pmol   = minao_mol( mol, minao='minao' )
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

    # Figure out which gaussian basis functions the vectors in ao2iao correspond to (mostly)
    num_iao   = ao2iao.shape[1] # Number of columns = # IAO
    gaussians = np.zeros( ao2iao.shape[0], dtype=int )
    for orb in range( num_iao ):
        maxval = np.max( ao2iao[:,orb] )
        minval = np.min( ao2iao[:,orb] )
        if ( abs( maxval ) > abs ( minval ) ):
            index = (( ao2iao[:,orb] == maxval ).argsort())[ len(gaussians)-1 ]
        else:
            index = (( ao2iao[:,orb] == minval ).argsort())[ len(gaussians)-1 ]
        gaussians[ index ] = 1
    assert( ao2iao.shape[1] == np.sum( gaussians ) )

    # Determine the complement of the IAO space
    DM_iao     = np.dot( ao2iao, ao2iao.T )
    mx         = np.dot( S1, np.dot( DM_iao, S1 ) )
    eigs, vecs = scipy.linalg.eigh( a=mx, b=S1 ) # Small to large in scipy
    ao2com     = vecs[ :, : Norbs - num_iao ]

    # Redo the IAO contruction for the complement space
    S31    = S1[  gaussians == 0 , : ]
    S3     = S31[ : , gaussians == 0 ]
    X      = np.linalg.solve( S3, np.dot( S31, ao2com ) )
    P13    = np.linalg.solve( S1, S31.T )
    Cp     = np.dot( P13, X )
    Cp     = orthogonalize_iao( Cp, S1 )
    DM1    = np.dot( ao2com, ao2com.T )
    DM3    = np.dot( Cp, Cp.T )
    A      = np.dot( DM1, np.dot( S1, np.dot( DM3, S31.T ) ) ) # All complement space is "occupied"...
    ao2com = orthogonalize_iao( A, S1 )

    if ( False ):
        # Localize the complement space with boys
        old_verbose = mol.verbose
        mol.verbose = 5
        loc         = localizer.localizer( mol, ao2com, 'boys', use_full_hessian=True )
        mol.verbose = old_verbose
        ao2com      = loc.optimize( threshold=1e-5 )
    ao2loc = np.hstack( ( ao2iao, ao2com ) )
    
    # Reorder the orbitals according to the atom list
    ao2loc = resort_orbitals( mol, ao2loc )
    
    # Check a few things:
    should_be_0 = np.dot( np.dot( ao2com.T, S1 ), ao2occ )
    should_be_1 = np.dot( np.dot( ao2loc.T, S1 ), ao2loc )
    print "QC-DMET :: iao_helper :: norm(     C_comp.T * S * C_occ  ) =", np.linalg.norm( should_be_0 )
    print "QC-DMET :: iao_helper :: norm( I - C_full.T * S * C_full ) =", np.linalg.norm( should_be_1 - np.eye( should_be_1.shape[0] ) )
    
    return ao2loc
    
    
