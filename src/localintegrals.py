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
sys.path.append('/home/seba') #Folder in which PySCF is installed
from pyscf import gto, scf, ao2mo, tools, future
from pyscf.future import lo
from pyscf.tools import molden, localizer
import rhf
import numpy as np

class localintegrals:

    def __init__( self, the_mf, active_orbs, localizationtype ):

        assert (( localizationtype == 'meta_lowdin' ) or ( localizationtype == 'boys' ))
        
        # Information on the full HF problem
        self.mol        = the_mf.mol
        self.fullEhf    = the_mf.hf_energy
        self.fullDMao   = np.dot(np.dot( the_mf.mo_coeff, np.diag( the_mf.mo_occ )), the_mf.mo_coeff.T )
        self.fullJKao   = scf.hf.get_veff( self.mol, self.fullDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.fullFOCKao = self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph') + self.fullJKao
        
        # Active space information
        self._which   = localizationtype
        self.active   = np.zeros( [ self.mol.nao_nr() ], dtype=int )
        self.active[ active_orbs ] = 1
        self.Norbs    = np.sum( self.active ) # Number of active space orbitals
        self.Nelec    = int(np.rint( self.mol.nelectron - np.sum( the_mf.mo_occ[ self.active==0 ] ))) # Total number of electrons minus frozen part
        
        # Localize the orbitals
        if ( self._which == 'meta_lowdin' ):
            assert( self.Norbs == self.mol.nao_nr() ) # Full active space required
            self.ao2loc = lo.orth.orth_ao( self.mol, self._which )
            self.TI_OK  = True
        if ( self._which == 'boys' ):
            loc = localizer.localizer( self.mol, the_mf.mo_coeff[ : , self.active==1 ], self._which )
            self.ao2loc = loc.optimize()
            self.TI_OK  = False
        assert( self.loc_ortho() < 1e-8 )
        
        # Effective Hamiltonian due to frozen part
        self.frozenDMmo  = np.array( the_mf.mo_occ, copy=True )
        self.frozenDMmo[ self.active==1 ] = 0 # Only the frozen MO occupancies nonzero
        self.frozenDMao  = np.dot(np.dot( the_mf.mo_coeff, np.diag( self.frozenDMmo )), the_mf.mo_coeff.T )
        self.frozenJKao  = scf.hf.get_veff( self.mol, self.frozenDMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.frozenOEIao = self.fullFOCKao - self.fullJKao + self.frozenJKao
        
        # Active space OEI and ERI
        self.activeCONST = self.mol.energy_nuc() + np.einsum( 'ij,ij->', self.frozenOEIao - 0.5*self.frozenJKao, self.frozenDMao )
        self.activeOEI   = np.dot( np.dot( self.ao2loc.T, self.frozenOEIao ), self.ao2loc )
        self.activeFOCK  = np.dot( np.dot( self.ao2loc.T, self.fullFOCKao  ), self.ao2loc )
        if ( self.Norbs <= 150 ):
            self.ERIinMEM  = True
            self.activeERI = ao2mo.outcore.full_iofree( self.mol, self.ao2loc, compact=False ).reshape(self.Norbs, self.Norbs, self.Norbs, self.Norbs)
        else:
            self.ERIinMEM  = False
            self.activeERI = None
        
        #self.debug_matrixelements()
        
    def molden( self, filename ):
    
        with open( filename, 'w' ) as thefile:
            molden.header( self.mol, thefile )
            molden.orbital_coeff( self.mol, thefile, self.ao2loc )
            
    def loc_ortho( self ):
    
        ShouldBeI = np.dot( np.dot( self.ao2loc.T , self.mol.intor('cint1e_ovlp_sph') ) , self.ao2loc )
        return np.linalg.norm( ShouldBeI - np.eye( ShouldBeI.shape[0] ) )
        
    def debug_matrixelements( self ):
    
        eigvals, eigvecs = np.linalg.eigh( self.activeFOCK )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        assert( self.Nelec % 2 == 0 )
        numPairs = self.Nelec / 2
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        if ( self.ERIinMEM == True ):
            DMloc = rhf.solve_ERI( self.activeOEI, self.activeERI, DMguess, numPairs )
        else:
            DMloc = rhf.solve_JK( self.activeOEI, self.mol, self.ao2loc, DMguess, numPairs )
        newFOCKloc = self.loc_rhf_fock_bis( DMloc )
        newRHFener = self.activeCONST + 0.5 * np.einsum( 'ij,ij->', DMloc, self.activeOEI + newFOCKloc )
        print "2-norm difference of RDM(self.activeFOCK) and RDM(self.active{OEI,ERI})  =", np.linalg.norm( DMguess - DMloc )
        print "2-norm difference of self.activeFOCK and FOCK(RDM(self.active{OEI,ERI})) =", np.linalg.norm( self.activeFOCK - newFOCKloc )
        print "RHF energy of mean-field input           =", self.fullEhf
        print "RHF energy based on self.active{OEI,ERI} =", newRHFener
        
    def exact_reference( self, method='ED', printstuff=True ):
    
        assert (( method == 'ED' ) or ( method == 'CC' ) or ( method == 'MP2' ))
        if ( self.ERIinMEM == False ):
            print "localintegrals::exact_reference : ERI of the localized orbitals are not stored in memory."
        assert ( self.ERIinMEM == True )
    
        print "Exact reference active space ( Norb, Nelec ) = (", self.Norbs, ",", self.Nelec, ")"
        chemical_pot = 0.0
        if ( method == 'ED' ):
            import chemps2
            GSenergy, GS_1DM = chemps2.solve( self.activeCONST, self.activeOEI, self.activeOEI, self.activeERI, self.Norbs, self.Nelec, self.Norbs, chemical_pot, printstuff )
        if ( method == 'CC' ):
            import psi4cc
            eigvals, eigvecs = np.linalg.eigh( self.activeFOCK )
            eigvecs = eigvecs[ :, eigvals.argsort() ]
            assert( self.Nelec % 2 == 0 )
            numPairs = self.Nelec / 2
            DMguessRHF = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
            GSenergy, GS_1DM = psi4cc.solve( self.activeCONST, self.activeOEI, self.activeOEI, self.activeERI, self.Norbs, self.Nelec, self.Norbs, DMguessRHF, chemical_pot, printstuff )
        if ( method == 'MP2' ):
            import pyscf_mp2
            eigvals, eigvecs = np.linalg.eigh( self.activeFOCK )
            eigvecs = eigvecs[ :, eigvals.argsort() ]
            assert( self.Nelec % 2 == 0 )
            numPairs = self.Nelec / 2
            DMguessRHF = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
            GSenergy, GS_1DM = pyscf_mp2.solve( self.activeCONST, self.activeOEI, self.activeOEI, self.activeERI, self.Norbs, self.Nelec, self.Norbs, DMguessRHF, chemical_pot, printstuff )
        print "Total",method,"ground state energy =", GSenergy
        return GSenergy
        
    def const( self ):
    
        return self.activeCONST
        
    def loc_oei( self ):
        
        return self.activeOEI
        
    def loc_rhf_fock( self ):
    
        return self.activeFOCK
        
    def loc_rhf_fock_bis( self, DMloc ):
    
        if ( self.ERIinMEM == False ):
            DM_ao = np.dot( np.dot( self.ao2loc, DMloc ), self.ao2loc.T )
            JK_ao = scf.hf.get_veff( self.mol, DM_ao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
            JK_loc = np.dot( np.dot( self.ao2loc.T, JK_ao ), self.ao2loc )
        else:
            JK_loc = np.einsum( 'ijkl,ij->kl', self.activeERI, DMloc ) - 0.5 * np.einsum( 'ijkl,ik->jl', self.activeERI, DMloc )
        FOCKloc = self.activeOEI + JK_loc
        return FOCKloc

    def loc_tei( self ):
    
        if ( self.ERIinMEM == False ):
            print "localintegrals::loc_tei : ERI of the localized orbitals are not stored in memory."
        assert ( self.ERIinMEM == True )
        return self.activeERI
        
    def dmet_oei( self, loc2dmet, numActive ):
    
        OEIdmet = np.dot( np.dot( loc2dmet[:,:numActive].T, self.activeOEI ), loc2dmet[:,:numActive] )
        return OEIdmet
        
    def dmet_fock( self, loc2dmet, numActive, coreDMloc ):
    
        FOCKdmet = np.dot( np.dot( loc2dmet[:,:numActive].T, self.loc_rhf_fock_bis( coreDMloc ) ), loc2dmet[:,:numActive] )
        return FOCKdmet
        
    def dmet_init_guess_rhf( self, loc2dmet, numActive, numPairs, Nimp, chempot_imp ):
    
        Fock_small = np.dot( np.dot( loc2dmet[:,:numActive].T, self.activeFOCK ), loc2dmet[:,:numActive] )
        if (chempot_imp != 0.0):
            for orb in range(Nimp):
                Fock_small[ orb, orb ] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh( Fock_small )
        eigvecs = eigvecs[ :, eigvals.argsort() ]
        DMguess = 2 * np.dot( eigvecs[ :, :numPairs ], eigvecs[ :, :numPairs ].T )
        return DMguess
        
    def dmet_tei( self, loc2dmet, numAct ):
    
        if ( self.ERIinMEM == False ):
            transfo = np.dot( self.ao2loc, loc2dmet[:,:numAct] )
            TEIdmet = ao2mo.outcore.full_iofree(self.mol, transfo, compact=False).reshape(numAct, numAct, numAct, numAct)
        else:
            TEIdmet = ao2mo.incore.full(ao2mo.restore(8, self.activeERI, self.Norbs), loc2dmet[:,:numAct], compact=False).reshape(numAct, numAct, numAct, numAct)
        return TEIdmet
        
        
