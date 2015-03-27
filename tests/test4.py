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
from pyscf import gto, scf
import numpy as np
import math as m

b1 = 1.2
b2 = 1.2
nmono = 10
mol = gto.Mole()
mol.verbose = 0
mol.atom = []

ang = 2*np.pi/nmono/2
c1 = b1 / np.cos(ang)
h = (c1+b2) / np.tan(ang)
r = np.sqrt(h**2+b2**2)
ang1 = np.arccos(h/r) * 2

for i in range(nmono):
    theta = i * 2*np.pi/nmono
    mol.atom.append(('H', (r*np.cos(theta),
                           r*np.sin(theta), 0)))
    mol.atom.append(('Li', (r*np.cos(ang1+theta),
                            r*np.sin(ang1+theta), 0)))

#note 2px 2py orbital do not have translational symmetry
mol.basis = {'H': '6-31g',
             'Li': gto.basis.parse('''
Li    S
    642.4189200              0.0021426        
     96.7985150              0.0162089        
     22.0911210              0.0773156        
      6.2010703              0.2457860        
      1.9351177              0.4701890        
      0.6367358              0.3454708        
Li    S
      2.3249184             -0.0350917
      0.6324306             -0.1912328
      0.0790534              1.0839878
Li    S
      0.0359620              1.0000000
''')}
mol.build()

mf = scf.RHF( mol )
mf.verbose = 3
mf.scf()

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.molden( 'qiming_lih.molden' )

#Imp size : 1 LiH
mono_per_imp = 1

impurityClusters = []
for cluster in range( nmono / mono_per_imp ):
    impurities = np.zeros( [ myInts.mol.nao_nr() ], dtype=int )
    for orb in range( 5*mono_per_imp ):
        impurities[ 5*mono_per_imp*cluster + orb ] = 1
    impurityClusters.append( impurities )
isTranslationInvariant = True
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant )
theDMET.doselfconsistent()


