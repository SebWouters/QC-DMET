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
sys.path.append('../src')
import localintegrals, dmet, qcdmet_paths
from pyscf import gto, scf, symm, future
from pyscf.future import cc
from pyscf.future.cc import ccsd
import numpy as np
import sn2_structures

'''
   thestructure can be
      * 'reactants_infinity'
      * 'reactants' (van der Waals bound ion)
      * any integer in the range [-9, 10] (boundaries included)
      * 'products'  (van der Waals bound ion)
      * 'products_infinity'
'''
thestructure = 0
thebasis1 = 'cc-pvdz'
thebasis2 = 'aug-cc-pvdz'

mol = sn2_structures.structure( thestructure, thebasis1, thebasis2 )

if (( True ) and ( 'infinity' not in str(thestructure) )):
    r_C  = np.array( mol.atom[0][1] )
    r_Cl = np.array( mol.atom[3][1] )
    r_Br = np.array( mol.atom[4][1] )
    dist_Cl_C = np.linalg.norm( r_C - r_Cl )
    dist_Br_C = np.linalg.norm( r_C - r_Br )
    print "Distance C - Cl =", dist_Cl_C
    print "Distance C - Br =", dist_Br_C

mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

if ( 'infinity' in str(thestructure) ):
    atom = sn2_structures.structure( thestructure.replace( 'infinity', 'atom' ), thebasis1, thebasis2 )
    mf_atom = scf.RHF( atom )
    mf_atom.verbose = 4
    mf_atom.scf()
    ccsolver = ccsd.CCSD( mf_atom )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    ERHF_extra  = mf_atom.hf_energy
    ECCSD_extra = mf_atom.hf_energy + ECORR
else:
    ERHF_extra  = 0.0
    ECCSD_extra = 0.0

if ( False ):
    from pyscf.tools import molden, localizer
    with open( 'sn2-mo.molden', 'w' ) as thefile:
        molden.header( mol, thefile )
        molden.orbital_coeff( mol, thefile, mf.mo_coeff )

if ( False ):
    ccsolver = ccsd.CCSD( mf )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    ECCSD = mf.hf_energy + ECORR
    print "ERHF  for structure", thestructure, "=", mf.hf_energy + ERHF_extra
    print "ECCSD for structure", thestructure, "=", ECCSD + ECCSD_extra
    
if ( True ):
    # myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'boys', localization_threshold=1e-5 )
    # myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
    myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'iao' )
    myInts.molden( 'sn2-loc.molden' )
    
    unit_sizes = None
    if (( thebasis1 == 'cc-pvdz' ) and ( thebasis2 == 'aug-cc-pvdz' )):
        if ( thestructure == 'reactants_infinity' ): # C12H25Br
            unit_sizes = np.array([ 60, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 29 ]) # Br, 11xCH2, CH3 (329 orbs total)
        elif ( thestructure == 'products_infinity' ): # C12H25Cl
            unit_sizes = np.array([ 51, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 29 ]) # Cl, 11xCH2, CH3 (320 orbs total)
        else:
            unit_sizes = np.array([ 87, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 29 ]) # ClBr, 11xCH2, CH3 (356 orbs total)
    assert( np.sum( unit_sizes ) == mol.nao_nr() )

    for carbons_in_cluster in range( 1, 7 ): #1,2,3,4,5,6
        orbs_in_imp = np.sum( unit_sizes[ 0 : carbons_in_cluster ] )
        impurityClusters = []
        impurities = np.zeros( [ mol.nao_nr() ], dtype=int )
        impurities[ 0 : orbs_in_imp ] = 1
        impurityClusters.append( impurities )

        theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant=False, method='CC', SCmethod='NONE' )
        theDMET.CC_E_TYPE = 'CASCI'
        theDMET.BATH_ORBS = 1 # Qiming, JCTC 10, 3784 (2014) [ http://dx.doi.org/10.1021/ct500512f ] for a C-C single bond
        the_energy = theDMET.doselfconsistent()
        print "######  DMET(", carbons_in_cluster,"C , CCSD ) /", thebasis1, "/", thebasis2, " =", the_energy + ECCSD_extra

    
