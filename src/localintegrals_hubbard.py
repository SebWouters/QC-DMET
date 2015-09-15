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

import rhf
import numpy as np

class localintegrals_hubbard:

    def __init__( self, hopping, HubbardU, Nelectrons ):
    
        self.activeOEI = hopping
        self.Norbs     = hopping.shape[0]
        self.Nelec     = Nelectrons
        self.HubbardU  = HubbardU
        
        eigvals, eigvecs = np.linalg.eigh( self.activeOEI )
        idx = eigvals.argsort()
        eigvals = eigvals[ idx ]
        eigvecs = eigvecs[ :, idx ]
        assert( Nelectrons % 2 == 0 )
        numPairs = Nelectrons / 2
        assert( eigvals[ numPairs ] - eigvals[ numPairs-1 ] > 1e-8 )
        #print "SP gap localint =", eigvals[ numPairs ] - eigvals[ numPairs-1 ]
        
        self.ao2loc = np.eye( self.Norbs )
        self.TI_OK  = True
        
        self.origDMloc  = 2 * np.dot( eigvecs[:,:numPairs], eigvecs[:,:numPairs].T )
        self.origJKloc  = np.zeros( [self.Norbs], dtype=float )
        for orb in range( self.Norbs ):
            self.origJKloc[ orb ] = 0.5 * self.HubbardU * self.origDMloc[ orb, orb ]
        self.origJKloc  = np.diag( self.origJKloc )
        self.fullEhf    = np.einsum( 'ij,ij->', self.activeOEI + 0.5 * self.origJKloc, self.origDMloc )
        self.activeFOCK = self.activeOEI + self.origJKloc
        
        print "Hubbard RHF energy =", self.fullEhf
        
        self.activeERI = np.zeros( [self.Norbs, self.Norbs, self.Norbs, self.Norbs], dtype=float)
        for orb in range( self.Norbs ):
            self.activeERI[ orb, orb, orb, orb ] = self.HubbardU
        self.ERIinMEM = True
        
    def const( self ):
    
        return 0.0
        
    def loc_oei( self ):
        
        return self.activeOEI
        
    def loc_rhf_fock( self ):
    
        return self.activeFOCK
        
    def loc_rhf_fock_bis( self, DMloc ):
    
        JKloc   = np.zeros( [self.Norbs], dtype=float )
        for orb in range( self.Norbs ):
            JKloc[ orb ] = 0.5 * self.HubbardU * DMloc[ orb, orb ]
        JKloc   = np.diag( JKloc )
        FOCKloc = self.activeOEI + JKloc
        return FOCKloc

    def loc_tei( self ):
    
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
    
        TEIdmet = np.zeros( [ numAct, numAct, numAct, numAct ] )
        for orb1 in range(numAct):
            for orb2 in range(numAct):
                for orb3 in range(numAct):
                    for orb4 in range(numAct):
                        value = 0.0
                        for orb in range( self.Norbs ):
                            value += loc2dmet[ orb, orb1 ] * loc2dmet[ orb, orb2 ] * loc2dmet[ orb, orb3 ] * loc2dmet[ orb, orb4 ]
                        TEIdmet[ orb1, orb2, orb3, orb4 ] = self.HubbardU * value
        return TEIdmet
        
