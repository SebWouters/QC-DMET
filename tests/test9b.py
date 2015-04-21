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

mol = gto.Mole() # C12H14 optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
    C           -6.789869536094    -0.280595658685     0.000000000000
    H           -7.730037326540     0.274233461510     0.000000000000
    H           -6.853933738256    -1.372796553918     0.000000000000
    C           -5.599775811061     0.350881201359     0.000000000000
    H           -5.578557362060     1.447468610830     0.000000000000
    C           -4.315159017926    -0.316235878929     0.000000000000
    H           -4.324681183597    -1.413140877795     0.000000000000
    C           -3.118052502019     0.329791975928     0.000000000000
    H           -3.114045962506     1.426988003011     0.000000000000
    C           -1.837863123066    -0.325130939213     0.000000000000
    H           -1.839256362924    -1.422071072736     0.000000000000
    C           -0.639643001594     0.326163860988     0.000000000000
    H           -0.639171522827     1.423200644980     0.000000000000
    C            0.639643001594    -0.326163860988     0.000000000000
    H            0.639171522827    -1.423200644980     0.000000000000
    C            1.837863123066     0.325130939213     0.000000000000
    H            1.839256362924     1.422071072736     0.000000000000
    C            3.118052502019    -0.329791975928     0.000000000000
    H            3.114045962506    -1.426988003011     0.000000000000
    C            4.315159017926     0.316235878929     0.000000000000
    H            4.324681183597     1.413140877795     0.000000000000
    C            5.599775811061    -0.350881201359     0.000000000000
    H            5.578557362060    -1.447468610830     0.000000000000
    C            6.789869536094     0.280595658685     0.000000000000
    H            7.730037326540    -0.274233461510     0.000000000000
    H            6.853933738256     1.372796553918     0.000000000000
  '''
mol.basis = '6-31G'
mol.symmetry = 0
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.molden( 'C12H14.molden' )

unit_sizes = np.array([ 13, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13 ]) # CH2, 10xCH, CH2
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


