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

mol = gto.Mole() # 1-chlorodecane from EMFT paper JCTC 11, 568-580 (2015)
mol.atom = '''
Cl            6.074014   -0.375527    0.000000
H             4.587791    1.234857   -0.888004
H             4.587791    1.234861    0.887998
C             4.535917    0.606269   -0.000001
H             3.322910   -0.929788   -0.878770
H             3.322911   -0.929785    0.878774
C             3.299764   -0.277657    0.000001
H             1.996337    1.211680    0.877653
H             1.996337    1.211677   -0.877656
C             2.008115    0.553293   -0.000001
H             0.753591   -0.969910   -0.876887
H             0.753592   -0.969908    0.876889
C             0.741064   -0.310560    0.000001
H            -0.565286    1.168692    0.876901
H            -0.565286    1.168693   -0.876900
C            -0.554527    0.508674    0.000000
H            -1.812063   -1.012283   -0.876749
H            -1.812062   -1.012285    0.876746
C            -1.823023   -0.352093    0.000000
H            -3.130077    1.126541    0.876796
H            -3.130077    1.126544   -0.876791
C            -3.119345    0.466230    0.000001
H            -4.378389   -1.054203   -0.876747
H            -4.378389   -1.054207    0.876742
C            -4.388205   -0.393603   -0.000001
H            -5.695612    1.083112    0.876169
H            -5.695612    1.083116   -0.876164
C            -5.685388    0.423431    0.000001
H            -6.982821   -1.090173   -0.882800
H            -7.853542    0.167696    0.000001
H            -6.982821   -1.090177    0.882796
C            -6.947283   -0.444115   -0.000001
'''
#mol.basis = '6-31g*'
mol.basis = 'sto-3g'
mol.symmetry = 0
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.molden( '1-chlorodecane.molden' )

if ( mol.basis == '6-31g*' ):
    unit_sizes = np.array([ 42, 24, 24, 24, 24, 24, 24, 24, 24, 29 ]) # 1 ClCH2, 8 CH2, and 1 CH3
if ( mol.basis == 'sto-3g' ):
    unit_sizes = np.array([ 16, 7, 7, 7, 7, 7, 7, 7, 7, 8 ]) # 1 ClCH2, 8 CH2, and 1 CH3
assert( np.sum( unit_sizes ) == mol.nao_nr() )

carbons_in_cluster = 2
units_counter = 0
orbitals_counter = 0

impurityClusters = []
while ( units_counter < len( unit_sizes ) ):
    impurities = np.zeros( [ mol.nao_nr() ], dtype=int )
    for unit in range( units_counter, min( len( unit_sizes ), units_counter + carbons_in_cluster ) ):
        impurities[ orbitals_counter : orbitals_counter + unit_sizes[ unit ] ] = 1
        orbitals_counter += unit_sizes[ unit ]
    units_counter += carbons_in_cluster
    impurityClusters.append( impurities )

totalcount = np.zeros( [ mol.nao_nr() ], dtype=int )
for item in impurityClusters:
    totalcount += item
assert ( np.linalg.norm( totalcount - np.ones( [ mol.nao_nr() ], dtype=float ) ) < 1e-12 )

method = 'CC'
isTranslationInvariant = False
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method )
theDMET.doselfconsistent()

