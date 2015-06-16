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

mol = gto.Mole() # C12H25Cl optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
    C            5.581691229271     0.699871241879     0.000000000000
    CL           7.128371256030    -0.266055445282     0.000000000000
    H            5.625540875759     1.338271613331     0.893773859693
    H            5.625540875759     1.338271613331    -0.893773859693
    C            4.353302072544    -0.197490969501     0.000000000000
    H            4.386671142451    -0.856130321847     0.884701861038
    H            4.386671142451    -0.856130321847    -0.884701861038
    C            3.051081608390     0.615098226633     0.000000000000
    H            3.031108062664     1.280554689463     0.883377834171
    H            3.031108062664     1.280554689463    -0.883377834171
    C            1.794855409289    -0.263463290760     0.000000000000
    H            1.816888734401    -0.929784149061     0.882559695773
    H            1.816888734401    -0.929784149061    -0.882559695773
    C            0.487744790263     0.536037506654     0.000000000000
    H            0.467526986958     1.202960588065     0.882604309802
    H            0.467526986958     1.202960588065    -0.882604309802
    C           -0.768607080943    -0.341365541129     0.000000000000
    H           -0.747556161490    -1.008486266399     0.882419699945
    H           -0.747556161490    -1.008486266399    -0.882419699945
    C           -2.077057338639     0.456180669319     0.000000000000
    H           -2.097737747030     1.123392593438     0.882453814358
    H           -2.097737747030     1.123392593438    -0.882453814358
    C           -3.333353545124    -0.421237154840     0.000000000000
    H           -3.312209557007    -1.088484884271     0.882411451597
    H           -3.312209557007    -1.088484884271    -0.882411451597
    C           -4.642182863738     0.375683564178     0.000000000000
    H           -4.662926834255     1.043004386837     0.882448246897
    H           -4.662926834255     1.043004386837    -0.882448246897
    C           -5.898577852071    -0.501203577286     0.000000000000
    H           -5.878543682564    -1.168716231737     0.882567899530
    H           -5.878543682564    -1.168716231737    -0.882567899530
    C           -7.207744089314     0.295405593377     0.000000000000
    H           -7.227479847004     0.961928078737     0.882187867638
    H           -7.227479847004     0.961928078737    -0.882187867638
    C           -8.456532873208    -0.589413169979     0.000000000000
    H           -8.484232690666    -1.242656894505     0.888980662405
    H           -8.484232690666    -1.242656894505    -0.888980662405
    H           -9.379750357996     0.012349306290     0.000000000000
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
myInts.molden( 'C12H25Cl.molden' )

if ( mol.basis == '6-31g' ):
    unit_sizes = np.array([ 26, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 15 ]) # CH2Cl, 10xCH2, CH3
if ( mol.basis == 'sto-3g' ):
    unit_sizes = np.array([ 16, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8 ]) # CH2Cl, 10xCH2, CH3
assert( np.sum( unit_sizes ) == mol.nao_nr() )

carbons_in_cluster = 1
impurityClusters = []
impurities = np.zeros( [ mol.nao_nr() ], dtype=int )
impurities[ : np.sum( unit_sizes[ : carbons_in_cluster ] ) ] = 1
impurityClusters.append( impurities )

isTranslationInvariant = False
method = 'CC'
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method )
theDMET.doselfconsistent()


