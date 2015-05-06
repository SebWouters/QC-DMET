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

mol = gto.Mole() # C12H26 optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
    C            7.037132908899     0.619261620324     0.000000000000
    H            7.957951524612     0.013771075858     0.000000000000
    H            7.067513108074     1.272431681336     0.888968291759
    H            7.067513108074     1.272431681336    -0.888968291759
    C            5.784660393975    -0.260379062096     0.000000000000
    H            5.801730350560    -0.927011488400     0.882170158970
    H            5.801730350560    -0.927011488400    -0.882170158970
    C            4.478664605024     0.541433093035     0.000000000000
    H            4.461407781270     1.209066926377     0.882562552477
    H            4.461407781270     1.209066926377    -0.882562552477
    C            3.218650782242    -0.330287082954     0.000000000000
    H            3.236813238940    -0.997750881699     0.882417237204
    H            3.236813238940    -0.997750881699    -0.882417237204
    C            1.912787576868     0.471533901404     0.000000000000
    H            1.894357112033     1.138948844606     0.882400782950
    H            1.894357112033     1.138948844606    -0.882400782950
    C            0.652961294601    -0.400871487371     0.000000000000
    H            0.671329825325    -1.068305377409     0.882385490399
    H            0.671329825325    -1.068305377409    -0.882385490399
    C           -0.652961294601     0.400871487371     0.000000000000
    H           -0.671329825325     1.068305377409     0.882385490399
    H           -0.671329825325     1.068305377409    -0.882385490399
    C           -1.912787576868    -0.471533901404     0.000000000000
    H           -1.894357112033    -1.138948844606     0.882400782950
    H           -1.894357112033    -1.138948844606    -0.882400782950
    C           -3.218650782242     0.330287082954     0.000000000000
    H           -3.236813238940     0.997750881699     0.882417237204
    H           -3.236813238940     0.997750881699    -0.882417237204
    C           -4.478664605024    -0.541433093035     0.000000000000
    H           -4.461407781270    -1.209066926377     0.882562552477
    H           -4.461407781270    -1.209066926377    -0.882562552477
    C           -5.784660393975     0.260379062096     0.000000000000
    H           -5.801730350560     0.927011488400     0.882170158970
    H           -5.801730350560     0.927011488400    -0.882170158970
    C           -7.037132908899    -0.619261620324     0.000000000000
    H           -7.067513108074    -1.272431681336     0.888968291759
    H           -7.067513108074    -1.272431681336    -0.888968291759
    H           -7.957951524612    -0.013771075858     0.000000000000
  '''
#mol.basis = '6-31g'
mol.basis = 'sto-3g'
mol.symmetry = 0
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.molden( 'C12H26.molden' )

if ( mol.basis == '6-31g' ):
    unit_sizes = np.array([ 15, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 15 ]) # CH3, 10xCH2, CH3
if ( mol.basis == 'sto-3g' ):
    unit_sizes = np.array([ 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8 ]) # CH3, 10xCH2, CH3
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

isTranslationInvariant = False
method = 'CC'
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method )
theDMET.doselfconsistent()


