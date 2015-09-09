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

mol = gto.Mole() # Caffeine optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
    C            1.817879727385     2.153638906169     0.000000000000
    C           -0.346505640687     3.462095703243     0.000000000000
    N            0.463691886339     2.246640891586     0.000000000000
    N            2.269654714050     0.899687380469     0.000000000000
    C           -0.003923667257     0.937835225180     0.000000000000
    C            1.134487063408     0.149351644916     0.000000000000
    C           -1.338251344864     0.403980438637     0.000000000000
    O           -2.385251021476     1.048239461893     0.000000000000
    N            1.058400655395    -1.225474274555     0.000000000000
    C            2.251397069015    -2.069224507359     0.000000000000
    C           -0.187094195596    -1.846487404954     0.000000000000
    O           -0.292432240842    -3.063478138223     0.000000000000
    N           -1.323645520078    -1.013034361391     0.000000000000
    C           -2.609849686479    -1.716498251901     0.000000000000
    H           -2.689209473523    -2.357766508547     0.889337124082
    H           -2.689209473523    -2.357766508547    -0.889337124082
    H           -3.397445032059    -0.956544015308     0.000000000000
    H            3.126343795339    -1.409688574359     0.000000000000
    H            2.260091440174    -2.714879611857    -0.890143668130
    H            2.260091440174    -2.714879611857     0.890143668130
    H            2.453380552599     3.037434448146     0.000000000000
    H           -1.400292735506     3.159575123448     0.000000000000
    H           -0.135202960256     4.062674697502     0.897532201407
    H           -0.135202960256     4.062674697502    -0.897532201407
  '''
mol.basis = 'cc-pvdz'
mol.symmetry = 1
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

mf = scf.RHF( mol )
mf.verbose = 3
mf.scf()

orbsym  = np.array(symm.label_orb_symm(mf.mol, mf.mol.irrep_id, mf.mol.symm_orb, mf.mo_coeff))
pi_orbs = (orbsym==1).nonzero()[0]
active  = np.hstack(( pi_orbs[:13], pi_orbs[16] ))

myInts = localintegrals.localintegrals( mf, active, 'boys' )
myInts.molden( 'caffeine.molden' )
method = 'MP2' # Method should be 'ED' or 'CC' or 'MP2'

impurityClusters = []
for cluster in range(len(active)):
    impurities = np.zeros( [ len(active) ], dtype=int )
    impurities[ cluster ] = 1
    impurityClusters.append( impurities )
isTranslationInvariant = False
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method )
theDMET.doselfconsistent()



