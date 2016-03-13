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

# C12H25Br + Cl-  -->  C12H25Cl + Br-

import sys
sys.path.append('../src')
import localintegrals, dmet, qcdmet_paths
from pyscf import gto, scf, symm, future
from pyscf.cc import ccsd
import numpy as np
import sn2_struct_bis

#############
#   Input   #
#############
thestructure = 0                    # 'reactants' or 'products' or any integer in the range [-9, 10] (boundaries included)
cluster_sizes = np.arange( 1, 7 )   # Number of carbon atoms per cluster
localization = 'iao'                # 'iao' or 'meta_lowdin' or 'boys'
single_impurity = True              # Single impurity vs. partitioning
one_bath_orb_per_bond = True        # Sun & Chan, JCTC 10, 3784 (2014) [ http://dx.doi.org/10.1021/ct500512f ]
casci_energy_formula = True         # CASCI or DMET energy formula

#######################
#   Parse the input   #
#######################
thebasis1 = 'cc-pvdz'       # Basis set for H and C
thebasis2 = 'aug-cc-pvdz'   # Basis set for Cl and Br
mol = sn2_structures.structure( thestructure, thebasis1, thebasis2 )
mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

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
    print "ERHF  for structure", thestructure, "=", mf.hf_energy
    print "ECCSD for structure", thestructure, "=", ECCSD
    
if ( True ):
    # myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'boys', localization_threshold=1e-5 )
    # myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
    myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'iao' )
    myInts.molden( 'sn2-loc.molden' )
    
    unit_sizes = None
    if (( thebasis1 == 'cc-pvdz' ) and ( thebasis2 == 'aug-cc-pvdz' )):
        unit_sizes = np.array([ 87, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 29 ]) # ClBr, 11xCH2, CH3 (356 orbs total)
    assert( np.sum( unit_sizes ) == mol.nao_nr() )

    for carbons_in_cluster in cluster_sizes:
        impurityClusters = []
        if ( casci_energy_formula ): # Do only 1 impurity at the edge
            num_orb_in_imp = np.sum( unit_sizes[ 0 : carbons_in_cluster ] )
            impurity_orbitals = np.zeros( [ mol.nao_nr() ], dtype=int )
            impurity_orbitals[ 0 : num_orb_in_imp ] = 1
            impurityClusters.append( impurity_orbitals )
        else: # Partition
            atoms_passed = 0
            jump = 0
            while ( atoms_passed < len( unit_sizes ) ):
                num_carb_in_imp = min( carbons_in_cluster, len( unit_sizes ) - atoms_passed )
                num_orb_in_imp = np.sum( unit_sizes[ atoms_passed : atoms_passed + num_carb_in_imp ] )
                impurity_orbitals = np.zeros( [ mol.nao_nr() ], dtype=int )
                if ( single_impurity and atoms_passed > 0 ):
                    impurity_orbitals[ jump : jump + num_orb_in_imp ] = -1
                else:
                    impurity_orbitals[ jump : jump + num_orb_in_imp ] = 1
                impurityClusters.append( impurity_orbitals )
                atoms_passed += num_carb_in_imp
                jump += num_orb_in_imp

        theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant=False, method='CC', SCmethod='NONE' )
        if ( casci_energy_formula == True ):
            theDMET.CC_E_TYPE = 'CASCI'
        if ( one_bath_orb_per_bond == True ):
            theDMET.BATH_ORBS = 2 * np.ones( [ len(impurityClusters) ], dtype=int )
            theDMET.BATH_ORBS[ 0 ] = 1
            theDMET.BATH_ORBS[ len(impurityClusters) - 1 ] = 1
        the_energy = theDMET.doselfconsistent()
        print "######  DMET(", carbons_in_cluster,"C , CCSD ) /", thebasis1, "/", thebasis2, " =", the_energy

    
