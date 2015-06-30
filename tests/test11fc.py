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
from pyscf import gto, scf, tools
from pyscf.tools import molden, localizer
import numpy as np
import math as m

factor = 1.0

b1 = 1.263 * factor
b2 = 1.132 * factor
nat = 16

sine  = np.sin( 2 * np.pi / nat )
cosin = np.cos( 2 * np.pi / nat )
R     = np.sqrt( 0.25 * b1 * b1 + ( ( b2 + b1 * cosin ) / ( 2 * sine ) ) ** 2 )
alpha = 2 * np.arcsin( 0.5 * b1 / R )

mol = gto.Mole()
mol.verbose = 5 # To print stuff during localization
mol.atom = []

angle = 0.0
for i in range( nat / 2 ):
    mol.atom.append(('C', (R * np.cos(angle        ), R * np.sin(angle        ), 0.0)))
    mol.atom.append(('C', (R * np.cos(angle + alpha), R * np.sin(angle + alpha), 0.0)))
    angle += 4.0 * np.pi / nat

mol.basis = 'sto-3g'
#mol.basis = 'cc-pvdz'
#mol.basis = 'cc-pvtz'
mol.build()

mf = scf.RHF( mol )
mf.verbose = 3
mf.scf()

#with open( 'george-rhf.molden', 'w' ) as thefile:
#    molden.header( mol, thefile )
#    molden.orbital_coeff( mol, thefile, mf.mo_coeff )

myInts = localintegrals.localintegrals( mf, range( nat, mol.nao_nr() ), 'boys' )
myInts.molden( 'george-fc.molden' )

atoms_per_imp = 1 # Impurity size = 1 C atom
assert ( nat % atoms_per_imp == 0 )
orbs_per_imp = myInts.Norbs * atoms_per_imp / nat

impurityClusters = []
for cluster in range( nat / atoms_per_imp ):
    impurities = np.zeros( [ myInts.Norbs ], dtype=int )
    for orb in range( orbs_per_imp ):
        impurities[ orbs_per_imp*cluster + orb ] = 1
    impurityClusters.append( impurities )
isTranslationInvariant = False # Because of Boys in C1
method = 'CC'
SCmethod = 'NONE' #Don't do it self-consistently
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod )
theDMET.doselfconsistent()


