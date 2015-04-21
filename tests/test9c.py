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

mol = gto.Mole() # C12H2 optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
    H            0.000000000000     0.000000000000    -8.144044918552
    C            0.000000000000     0.000000000000    -7.071398867071
    C            0.000000000000     0.000000000000    -5.849562285786
    C            0.000000000000     0.000000000000    -4.490828010798
    C            0.000000000000     0.000000000000    -3.256676356402
    C            0.000000000000     0.000000000000    -1.910115133978
    C            0.000000000000     0.000000000000    -0.672009763510
    C            0.000000000000     0.000000000000     0.672009763510
    C            0.000000000000     0.000000000000     1.910115133978
    C            0.000000000000     0.000000000000     3.256676356402
    C            0.000000000000     0.000000000000     4.490828010798
    C            0.000000000000     0.000000000000     5.849562285786
    C            0.000000000000     0.000000000000     7.071398867071
    H            0.000000000000     0.000000000000     8.144044918552
  '''
mol.basis = 'cc-pVDZ'
mol.symmetry = 0
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.molden( 'C12H2.molden' )

unit_sizes = np.array([ 19, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 19 ]) # CH, 10xC, CH
assert( np.sum( unit_sizes ) == mol.nao_nr() )

method = 'CC'
myInts.exact_reference( method, True ) # Printing

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

isTranslationInvariant = False
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method )
theDMET.doselfconsistent()


