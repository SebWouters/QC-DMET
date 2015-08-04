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
from pyscf import gto, scf, ao2mo
import numpy as np

DMguess  = None
#old_umat = None

bondlengths = np.arange(0.6, 3.05, 0.1)
#bondlengths = np.arange(0.6, 3.05, 0.1)[::-1]
energies = []

for bondlength in bondlengths:

    nat = 10
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bondlength / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('H', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = 'sto-3g'
    mol.build(verbose=0)

    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=DMguess)

    if ( False ):   
        ENUCL = mf.mol.energy_nuc()
        OEI   = np.dot(np.dot(mf.mo_coeff.T, mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')), mf.mo_coeff)
        TEI   = ao2mo.outcore.full_iofree(mol, mf.mo_coeff, compact=False).reshape(mol.nao_nr(), mol.nao_nr(), mol.nao_nr(), mol.nao_nr())
        import chemps2
        Energy, OneDM = chemps2.solve( ENUCL, OEI, OEI, TEI, mol.nao_nr(), mol.nelectron, mol.nao_nr(), 0.0, False )
        print "bl =", bondlength," and energy =", Energy
        
    else:
        myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), 'meta_lowdin' )
        myInts.molden( 'hydrogen-loc.molden' )

        atoms_per_imp = 2 # Impurity size = 1 atom
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
        SCmethod = 'LSTSQ' #Don't do it self-consistently
        doSCF = False
        theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod, doSCF )
        #if ( old_umat != None ):
        #    theDMET.umat = np.array( old_umat, copy=True )
        Energy = theDMET.doselfconsistent()
        #old_umat = np.array( theDMET.umat, copy=True )
        print "bl =", bondlength," and energy =", Energy
        energies.append(Energy)
        #theDMET.dump_bath_orbs( 'hydrogen-bath.molden' )
        #DMguess = np.dot( np.dot( myInts.ao2loc, theDMET.onedm_solution_rhf() ), myInts.ao2loc.T )

print "Bondlengths =", bondlengths
print "Energies =", energies

