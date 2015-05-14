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
sys.path.append('/home/seba/psi4cc')
sys.path.append('/tigress/sebastian.wouters/sw-della/psi4cc')
import psi4
import rhf
import localintegrals
from pyscf import ao2mo

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF, chempot_imp=0.0, printoutput=False ):

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
    assert( Nel % 2 == 0 )
    numPairs = Nel / 2
    DMrhf   = rhf.solve_ERI( FOCKcopy, TEI, DMguessRHF, numPairs )
    Erhf    = CONST + np.einsum('ij,ij->', FOCKcopy, DMrhf)
    Erhf   += 0.5 * np.einsum('ijkl,ij,kl->', TEI, DMrhf, DMrhf) - 0.25 * np.einsum('ijkl,ik,jl->', TEI, DMrhf, DMrhf)
    
    # Rotate everything to the MO basis
    FOCKrhf = FOCKcopy + np.einsum('ijkl,ij->kl', TEI, DMrhf) - 0.5 * np.einsum('ijkl,ik->jl', TEI, DMrhf)
    eigvals, eigvecs = np.linalg.eigh( FOCKrhf )
    idx = eigvals.argsort()
    eigvals = eigvals[ idx ]
    print "psi4cc::solve : RHF homo-lumo gap =", eigvals[numPairs] - eigvals[numPairs-1]
    eigvecs = eigvecs[ :, idx ]
    DMrhf2  = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
    print "Two-norm difference of 1-RDM(RHF) and 1-RDM(FOCK(RHF)) =", np.linalg.norm(DMrhf - DMrhf2)
    FOCKcopy_mo = np.dot( eigvecs.T, np.dot( FOCKcopy, eigvecs ) )
    TEI_mo      = ao2mo.incore.full( ao2mo.restore(8, TEI, Norb), eigvecs, compact=False ).reshape(Norb, Norb, Norb, Norb)
    
    # Get the CC solution
    psi4cc = psi4.Solver( (1<<33) ) # Standard in psi4cc was (1<<30)=1.07Gb. (1<<33)=8.59Gb.
    psi4cc.prepare('RHF', np.eye(Norb), FOCKcopy_mo, ao2mo.restore(4,TEI_mo,Norb), Nel)
    Ecorrelation = psi4cc.energy('CCSD')
    OneRDM_mo, TwoRDM_mo = psi4cc.density() # 2-RDM is stored in physics notation!
    OneRDM_mo = OneRDM_mo + OneRDM_mo.T # Symmetrize the CC 1-RDM and adjust the missing factor of two!
    TwoRDM_mo = np.swapaxes( TwoRDM_mo, 1, 2 ) # Now TwoRDM is stored in chemists notation!
    del psi4cc
    
    # Check that we understand what is going on
    Etotal = Erhf + Ecorrelation
    for orb1 in range(numPairs):
        OneRDM_mo[orb1, orb1] += 2.0
        for orb2 in range(numPairs):
            TwoRDM_mo[orb1,orb1,orb2,orb2] += 4.0
            TwoRDM_mo[orb1,orb2,orb1,orb2] -= 2.0
    OneRDM_loc = np.dot(eigvecs, np.dot( OneRDM_mo, eigvecs.T ))
    TwoRDM_loc = np.einsum('ai,ijkl->ajkl', eigvecs, TwoRDM_mo )
    TwoRDM_loc = np.einsum('bj,ajkl->abkl', eigvecs, TwoRDM_loc)
    TwoRDM_loc = np.einsum('ck,abkl->abcl', eigvecs, TwoRDM_loc)
    TwoRDM_loc = np.einsum('dl,abcl->abcd', eigvecs, TwoRDM_loc)
    Etotal2 = CONST + np.einsum('ij,ij->', FOCKcopy, OneRDM_loc) + 0.5 * np.einsum('ijkl,ijkl->', TEI, TwoRDM_loc)
    print "Etotal  =", Etotal
    print "Etotal2 =", Etotal2
    
    # Reviving output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)
    
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', OneRDM_loc[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] )
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:Nimp,:,:], TEI[:,:Nimp,:,:] )
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:Nimp,:], TEI[:,:,:Nimp,:] )
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:,:Nimp], TEI[:,:,:,:Nimp] )
    return ( ImpurityEnergy, OneRDM_loc )

