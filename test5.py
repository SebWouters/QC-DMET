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
sys.path.append('src')
import localintegrals, dmet
from pyscf import gto
import numpy as np

mol = gto.Mole() # Benzene optimized with Psi4 B3LYP/cc-pVDZ
mol.atom = '''
     H    0.000000000000     2.491406946734     0.000000000000
     C    0.000000000000     1.398696930758     0.000000000000
     H    0.000000000000    -2.491406946734     0.000000000000
     C    0.000000000000    -1.398696930758     0.000000000000
     H    2.157597486829     1.245660462400     0.000000000000
     C    1.211265339156     0.699329968382     0.000000000000
     H    2.157597486829    -1.245660462400     0.000000000000
     C    1.211265339156    -0.699329968382     0.000000000000
     H   -2.157597486829     1.245660462400     0.000000000000
     C   -1.211265339156     0.699329968382     0.000000000000
     H   -2.157597486829    -1.245660462400     0.000000000000
     C   -1.211265339156    -0.699329968382     0.000000000000
  '''
mol.basis = 'sto-6g'
mol.symmetry = 0
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

myInts = localintegrals.localintegrals( mol )
myInts.molden( 'benzene.molden' )

impurityClusters = []
for cluster in range(6):
    impurities = np.zeros( [ myInts.mol.nao_nr() ], dtype=int )
    for orb in range(6):
        impurities[ 6*cluster + orb ] = 1
    impurityClusters.append( impurities )
isTranslationInvariant = False
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant )
theDMET.doselfconsistent()



