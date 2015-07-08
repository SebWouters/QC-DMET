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

bl = 1.0
nat = 24
mol = gto.Mole()
mol.atom = []
r = 0.5 * bl / np.sin(np.pi/nat)
for i in range(nat):
    theta = i * (2*np.pi/nat)
    mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

mol.basis = 'sto-3g'
mol.build(verbose=0)

mf = scf.RHF(mol)
mf.verbose = 3
mf.max_cycle = 1000
mf.scf()

myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
myInts.molden( 'Hnew-loc.molden' )

atoms_per_imp = 2 # Impurity size = 1/2/3/4/6/8 Be atoms
assert ( nat % atoms_per_imp == 0 )
orbs_per_imp = myInts.Norbs * atoms_per_imp / nat

impurityClusters = []
for cluster in range( nat / atoms_per_imp ):
    impurities = np.zeros( [ myInts.Norbs ], dtype=int )
    for orb in range( orbs_per_imp ):
        impurities[ orbs_per_imp*cluster + orb ] = 1
    impurityClusters.append( impurities )
isTranslationInvariant = True
method = 'ED'
SCmethod = 'HUBB' #'LSTSQ'
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod )
theDMET.doselfconsistent()
theDMET.dump_bath_orbs( 'Hnew-bathorbs.molden' )


