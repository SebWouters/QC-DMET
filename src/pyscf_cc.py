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

import numpy as np
import ctypes
import os
import sys
from pyscf import ao2mo, gto, scf, future
from pyscf.future import cc
from pyscf.future.cc import ccsd
from pyscf.tools import rhf_newtonraphson

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF, energytype='LAMBDA', chempot_imp=0.0, printoutput=True ):

    assert (( energytype == 'LAMBDA' ) or ( energytype == 'LAMBDA_AMP' ) or ( energytype == 'LAMBDA_ZERO' ))

    # Killing output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)

    # Augment the FOCK operator with the chemical potential
    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp
    
    # Get the RHF solution
    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('C', (0, 0, 0)))
    mol.nelectron = Nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf( DMguessRHF )
    DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    if ( mf.converged == False ):
        mf = rhf_newtonraphson.solve( mf, dm_guess=DMloc )
        DMloc = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    ERHF1 = mf.hf_energy
    
    # Check the RHF solution
    assert( Nel % 2 == 0 )
    numPairs = Nel / 2
    FOCKloc = FOCKcopy + np.einsum('ijkl,ij->kl', TEI, DMloc) - 0.5 * np.einsum('ijkl,ik->jl', TEI, DMloc)
    eigvals, eigvecs = np.linalg.eigh( FOCKloc )
    idx = eigvals.argsort()
    eigvals = eigvals[ idx ]
    eigvecs = eigvecs[ :, idx ]
    print "psi4cc::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs-1]
    DMloc2  = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
    print "Two-norm difference of 1-RDM(RHF) and 1-RDM(FOCK(RHF)) =", np.linalg.norm(DMloc - DMloc2)
    
    # Get the CC solution from pyscf
    ccsolver = ccsd.CCSD( mf )
    ccsolver.verbose = 5
    ECORR, t1, t2 = ccsolver.ccsd()
    if ( energytype == 'LAMBDA' ):
        ccsolver.solve_lambda()
        pyscfRDM1 = ccsolver.make_rdm1() # MO space
        pyscfRDM2 = ccsolver.make_rdm2() # MO space
    if ( energytype == 'LAMBDA_AMP' ):
        # Overwrite lambda tensors with t-amplitudes
        pyscfRDM1 = ccsolver.make_rdm1(t1, t2, t1, t2) # MO space
        pyscfRDM2 = ccsolver.make_rdm2(t1, t2, t1, t2) # MO space
    if ( energytype == 'LAMBDA_ZERO' ):
        # Overwrite lambda tensors with 0.0
        fake_l1 = np.zeros( t1.shape, dtype=float )
        fake_l2 = np.zeros( t2.shape, dtype=float )
        pyscfRDM1 = ccsolver.make_rdm1(t1, t2, fake_l1, fake_l2) # MO space
        pyscfRDM2 = ccsolver.make_rdm2(t1, t2, fake_l1, fake_l2) # MO space
    pyscfRDM1 = 0.5 * ( pyscfRDM1 + pyscfRDM1.T ) # Symmetrize
    ERHF2 = mf.hf_energy
    
    # Print a few to things to double check
    print "ERHF1 =", ERHF1
    print "ERHF2 =", ERHF2
    '''
    print "Do we understand how the 1-RDM is stored?", np.linalg.norm( np.einsum('ii->',     pyscfRDM1) - Nel )
    print "Do we understand how the 2-RDM is stored?", np.linalg.norm( np.einsum('ijkk->ij', pyscfRDM2) / (Nel - 1.0) - pyscfRDM1 )
    '''
    ECCSD1 = ERHF2 + ECORR
    # Change the pyscfRDM1/2 from MO space to localized space
    pyscfRDM1 = np.dot(mf.mo_coeff, np.dot(pyscfRDM1, mf.mo_coeff.T ))
    pyscfRDM2 = np.einsum('ai,ijkl->ajkl', mf.mo_coeff, pyscfRDM2)
    pyscfRDM2 = np.einsum('bj,ajkl->abkl', mf.mo_coeff, pyscfRDM2)
    pyscfRDM2 = np.einsum('ck,abkl->abcl', mf.mo_coeff, pyscfRDM2)
    pyscfRDM2 = np.einsum('dl,abcl->abcd', mf.mo_coeff, pyscfRDM2)
    ECCSD2 = CONST + np.einsum('ij,ij->', FOCKcopy, pyscfRDM1) + 0.5 * np.einsum('ijkl,ijkl->', TEI, pyscfRDM2)
    print "ECCSD1 =", ECCSD1
    print "ECCSD2 =", ECCSD2
    
    # Reviving output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)
    
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    ImpurityEnergy = CONST \
                   + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:Nimp,:],     FOCK[:Nimp,:] + OEI[:Nimp,:]) \
                   + 0.25  * np.einsum('ij,ij->',     pyscfRDM1[:,:Nimp],     FOCK[:,:Nimp] + OEI[:,:Nimp]) \
                   + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:Nimp,:,:,:], TEI[:Nimp,:,:,:]) \
                   + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:Nimp,:,:], TEI[:,:Nimp,:,:]) \
                   + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:Nimp,:], TEI[:,:,:Nimp,:]) \
                   + 0.125 * np.einsum('ijkl,ijkl->', pyscfRDM2[:,:,:,:Nimp], TEI[:,:,:,:Nimp])
    
    return ( ImpurityEnergy, pyscfRDM1 )

