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
import rhf
import localintegrals
import os  # for dev/null
import sys # for sys.stdout
import qcdmet_paths
from pyscf import gto, scf, ao2mo, mp

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, DMguessRHF, chempot_imp=0.0, printoutput=True ):

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
    mol = gto.Mole()
    mol.build(verbose=0)
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = Nel
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf( DMguessRHF )
    DMrhf = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
    Erhf  = CONST + np.einsum('ij,ij->', FOCKcopy, DMrhf)
    Erhf += 0.5 * np.einsum('ijkl,ij,kl->', TEI, DMrhf, DMrhf) - 0.25 * np.einsum('ijkl,ik,jl->', TEI, DMrhf, DMrhf)
    numPairs = Nel / 2
    print "pyscf_mp2::solve : RHF homo-lumo gap =", mf.mo_energy[numPairs] - mf.mo_energy[numPairs-1]
    
    # Get the MP2 solution
    myMP2 = mp.MP2( mf )
    E_MP2, T_MP2 = myMP2.kernel()
    OneRDM_mo = np.zeros( [Norb, Norb], dtype=float )
    #OneRDM_mo = myMP2.make_rdm1() --------------> Carlos his remark
    TwoRDM_mo = myMP2.make_rdm2() # 2-RDM is stored in chemistry notation!
    #OneRDM_mo = 0.5 * ( OneRDM_mo + OneRDM_mo.T ) # Symmetrize
    
    # Check that we understand what is going on
    Etotal = Erhf + E_MP2
    for orb1 in range(numPairs):
        OneRDM_mo[orb1, orb1] += 2.0
        for orb2 in range(numPairs):
            TwoRDM_mo[orb1,orb1,orb2,orb2] += 4.0
            TwoRDM_mo[orb1,orb2,orb1,orb2] -= 2.0
    OneRDM_loc = np.dot(mf.mo_coeff, np.dot( OneRDM_mo, mf.mo_coeff.T ))
    TwoRDM_loc = np.einsum('ai,ijkl->ajkl', mf.mo_coeff, TwoRDM_mo )
    TwoRDM_loc = np.einsum('bj,ajkl->abkl', mf.mo_coeff, TwoRDM_loc)
    TwoRDM_loc = np.einsum('ck,abkl->abcl', mf.mo_coeff, TwoRDM_loc)
    TwoRDM_loc = np.einsum('dl,abcl->abcd', mf.mo_coeff, TwoRDM_loc)
    Etotal2 = Erhf - 0.5 * np.einsum('ijkl,ij,kl->', TEI, DMrhf, DMrhf) + 0.25 * np.einsum('ijkl,ik,jl->', TEI, DMrhf, DMrhf) + 0.5 * np.einsum('ijkl,ijkl->', TEI, TwoRDM_loc)
    print "Etotal  =", Etotal
    print "Etotal2 =", Etotal2
    
    # Reviving output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)
    
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', DMrhf[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] ) # To be consistent with the energy formula above, this should be the HF RDM !!!
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:Nimp,:,:], TEI[:,:Nimp,:,:] )
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:Nimp,:], TEI[:,:,:Nimp,:] )
    ImpurityEnergy += 0.125 * np.einsum( 'ijkl,ijkl->', TwoRDM_loc[:,:,:,:Nimp], TEI[:,:,:,:Nimp] )
    return ( ImpurityEnergy, OneRDM_loc )

