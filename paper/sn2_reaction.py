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
import localintegrals, dmet
from pyscf import gto, scf, symm, future
from pyscf.future import cc
from pyscf.future.cc import ccsd
import numpy as np
import sn2_structures

thestructure = 0
thebasis = 'cc-pvdz'

mol = sn2_structures.structure( thestructure, thebasis )
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

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'boys', localization_threshold=1e-5 )
myInts.molden( 'sn2-loc.molden' )

unit_sizes = None
if ( mol.basis == 'cc-pvdz' ):
    unit_sizes = np.array([ 69, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 29 ]) # CH2ClBr, 10xCH2, CH3 (338 orbs total)
assert( np.sum( unit_sizes ) == mol.nao_nr() )

for carbons_in_cluster in range( 1, 5 ): #1,2,3,4
    orbs_in_imp = np.sum( unit_sizes[ 0 : carbons_in_cluster ] )
    impurityClusters = []
    impurities = np.zeros( [ mol.nao_nr() ], dtype=int )
    impurities[ 0 : orbs_in_imp ] = 1
    impurityClusters.append( impurities )
    
    isTranslationInvariant = False
    method = 'CC'
    SCmethod = 'NONE' # <--- because only 1 impurity in large HF environment
    theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod )
    the_energy = theDMET.doselfconsistent()
    print "######  DMET(", carbons_in_cluster," C , CCSD ) /", thebasis, " =", the_energy


