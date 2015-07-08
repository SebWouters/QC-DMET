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
from pyscf import gto, scf, symm
import numpy as np

mol = gto.Mole() # Buckyball optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
 C    -2.53113999    0.00000000    0.34113867
 H    -2.53149095   -0.89054959    0.96946592
 H    -2.53149095    0.89054959    0.96946592
 H    -3.42112071    0.00000000   -0.28817303
 C    -1.26556968    0.00000000   -0.55375339
 H    -1.26592020   -0.88867069   -1.18404849
 H    -1.26592020    0.88867069   -1.18404849
 C     0.00000000   -0.00000000    0.34113956
 H     0.00000000   -0.89144119    0.96960776
 H    -0.00000000    0.89144119    0.96960776
 C     1.26556968    0.00000000   -0.55375339
 H     1.26592020    0.88867069   -1.18404849
 H     1.26592020   -0.88867069   -1.18404849
 C     2.53113999    0.00000000    0.34113867
 H     2.53149095   -0.89054959    0.96946592
 H     3.42112071    0.00000000   -0.28817303
 H     2.53149095    0.89054959    0.96946592
 '''
mol.basis = 'cc-pvdz'
mol.build(verbose=0)

mf = scf.RHF(mol)
mf.verbose = 3
mf.scf()

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'lowdin' )
myInts.molden( 'Pentane-loc.molden' )

imp_sizes = np.array( [ 29, 24, 24, 24, 29 ] )
assert ( np.sum( imp_sizes ) == myInts.Norbs )

impurityClusters = []
jump = 0
for item in imp_sizes:
    impurities = np.zeros( [ myInts.Norbs ], dtype=int )
    impurities[ jump : jump + item ] = 1
    impurityClusters.append( impurities )
    jump += item
isTranslationInvariant = False
method = 'MP2'
SCmethod = 'NONE'
doSCF = False
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod, doSCF )
theDMET.doselfconsistent()
for imp_number in range(5):
    theDMET.dump_bath_orbs( 'Pentane-bathorbs-imp'+str(imp_number)+'.molden', imp_number )


