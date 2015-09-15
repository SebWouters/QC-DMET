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
import os  # for dev/null
import sys # for sys.stdout
import qcdmet_paths
import PyCheMPS2

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, chempot_imp=0.0, printoutput=False ):

    # Set the seed of the random number generator and cout.precision
    Initializer = PyCheMPS2.PyInitialize()
    Initializer.Init()

    # Setting up the Hamiltonian
    Group = 0
    orbirreps = np.zeros([ Norb ], dtype=ctypes.c_int)
    HamCheMPS2 = PyCheMPS2.PyHamiltonian(Norb, Group, orbirreps)
    HamCheMPS2.setEconst( CONST )
    for cnt1 in range(Norb):
        for cnt2 in range(Norb):
            HamCheMPS2.setTmat(cnt1, cnt2, FOCK[cnt1, cnt2])
            for cnt3 in range(Norb):
                for cnt4 in range(Norb):
                    HamCheMPS2.setVmat(cnt1, cnt2, cnt3, cnt4, TEI[cnt1, cnt3, cnt2, cnt4]) #From chemist to physics notation
    '''HamCheMPS2.save()
    exit(123)'''
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            HamCheMPS2.setTmat(orb, orb, FOCK[orb, orb] - chempot_imp)
    
    # Killing output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)
        
    if ( Norb <= 10 ):
    
        # FCI ground state calculation
        assert( Nel % 2 == 0 )
        Nel_up       = Nel / 2
        Nel_down     = Nel / 2
        Irrep        = 0
        maxMemWorkMB = 100.0
        FCIverbose   = 2
        theFCI = PyCheMPS2.PyFCI( HamCheMPS2, Nel_up, Nel_down, Irrep, maxMemWorkMB, FCIverbose )
        GSvector = np.zeros( [ theFCI.getVecLength() ], dtype=ctypes.c_double )
        theFCI.FillRandom( theFCI.getVecLength() , GSvector ) # Random numbers in [-1,1[
        GSvector[ theFCI.LowestEnergyDeterminant() ] = 12.345 # Large component for quantum chemistry
        EnergyCheMPS2 = theFCI.GSDavidson( GSvector )
        #SpinSquared = theFCI.CalcSpinSquared( GSvector )
        TwoRDM = np.zeros( [ Norb**4 ], dtype=ctypes.c_double )
        theFCI.Fill2RDM( GSvector, TwoRDM )
        TwoRDM = TwoRDM.reshape( [Norb, Norb, Norb, Norb], order='F' )
        TwoRDM = np.swapaxes( TwoRDM, 1, 2 ) #From physics to chemistry notation
        del theFCI
        
    else:
    
        # DMRG ground state calculation
        assert( Nel % 2 == 0 )
        TwoS  = 0
        Irrep = 0
        Prob  = PyCheMPS2.PyProblem( HamCheMPS2, TwoS, Nel, Irrep )

        OptScheme = PyCheMPS2.PyConvergenceScheme(3) # 3 instructions
        #OptScheme.setInstruction(instruction, D, Econst, maxSweeps, noisePrefactor)
        OptScheme.setInstruction(0,  500, 1e-10,  3, 0.05)
        OptScheme.setInstruction(1, 1000, 1e-10,  3, 0.05)
        OptScheme.setInstruction(2, 1000, 1e-10, 10, 0.00) # Last instruction a few iterations without noise

        theDMRG = PyCheMPS2.PyDMRG( Prob, OptScheme )
        EnergyCheMPS2 = theDMRG.Solve()
        theDMRG.calc2DMandCorrelations()
        TwoRDM = np.zeros( [Norb, Norb, Norb, Norb], dtype=ctypes.c_double )
        for orb1 in range(Norb):
            for orb2 in range(Norb):
                for orb3 in range(Norb):
                    for orb4 in range(Norb):
                        TwoRDM[ orb1, orb3, orb2, orb4 ] = theDMRG.get2DMA( orb1, orb2, orb3, orb4 ) #From physics to chemistry notation

        # theDMRG.deleteStoredMPS()
        theDMRG.deleteStoredOperators()
        del theDMRG
        del OptScheme
        del Prob

    del HamCheMPS2

    # Reviving output if necessary
    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)
    
    # Calculate the energy of the impurity based on the 1RDM and 2RDM
    OneRDM = np.einsum( 'ijkk->ij', TwoRDM ) / ( Nel - 1 )
    
    # To calculate the impurity energy, rescale the JK matrix with a factor 0.5 to avoid double counting: 0.5 * ( OEI + FOCK ) = OEI + 0.5 * JK
    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', OneRDM[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] )
    ImpurityEnergy += 0.5 * np.einsum( 'ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )
    return ( ImpurityEnergy, OneRDM )

