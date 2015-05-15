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

mol = gto.Mole()  # 1-decanol from EMFT paper JCTC 11, 568-580 (2015)
mol.atom = '''
O        -6.326993   -0.342551   -0.085075
H        -6.343597   -0.932869    0.674603
H        -5.172280    1.115811   -0.856164
H        -5.163379    1.078922    0.909461
C        -5.143976    0.447568    0.008436
H        -3.870394   -1.068468    0.848317
H        -3.886864   -1.023249   -0.909394
C        -3.866782   -0.389712   -0.015871
H        -2.586810    1.131129   -0.857112
H        -2.589011    1.097064    0.896328
C        -2.587307    0.454246    0.006651
H        -1.301618   -1.058312    0.856288
H        -1.301459   -1.025748   -0.896865
C        -1.302629   -0.382041   -0.008161
H        -0.024656    1.137733   -0.856268
H        -0.024980    1.105335    0.896879
C        -0.022767    0.461248    0.008048
H         1.263962   -1.051267    0.856601
H         1.264619   -1.017717   -0.896424
C         1.262430   -0.374150   -0.007297
H         2.541053    1.146168   -0.854028
H         2.540990    1.111571    0.899047
C         2.542888    0.468512    0.009449
H         3.830289   -1.045369    0.855777
H         3.831311   -1.009068   -0.897251
C         3.828011   -0.366637   -0.007063
H         5.107586    1.152957   -0.850500
H         5.107033    1.115433    0.901323
C         5.109507    0.474593    0.011259
H         6.434418   -1.033802    0.861582
H         7.282187    0.259647    0.008368
H         6.435992   -0.994161   -0.903519
C         6.387550   -0.368849   -0.006524
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
myInts.molden( '1-decanol.molden' )

if ( mol.basis == '6-31g*' ):
    unit_sizes = np.array([ 43, 24, 24, 24, 24, 24, 24, 24, 24, 29 ]) # 1 CH2OH, 8 CH2, and 1 CH3
if ( mol.basis == 'sto-3g' ):
    unit_sizes = np.array([ 13, 7, 7, 7, 7, 7, 7, 7, 7, 8 ]) # 1 CH2OH, 8 CH2, and 1 CH3
assert( np.sum( unit_sizes ) == mol.nao_nr() )

carbons_in_cluster = 1
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

