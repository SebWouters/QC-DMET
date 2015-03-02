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
from pyscf.tools import molden
import numpy as np

class localintegrals:

    def __init__( self, molecule ):
    
        self.mol    = molecule
        self.ao2loc = lo.orth.orth_ao( self.mol, 'meta_lowdin' )
        assert( self.loc_ortho() < 1e-8 )
        mf = scf.RHF( self.mol )
        mf.verbose = 0
        mf.scf()
        DMao = np.dot(np.dot( mf.mo_coeff, np.diag( mf.mo_occ )), mf.mo_coeff.T )
        self.mf_energy = mf.hf_energy
        self.JKao      = scf.hf.get_veff( self.mol, DMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        self.eri       = mf._eri
        self.Norbs     = self.mol.nao_nr()
        self.Nelec     = self.mol.nelectron
        
    def molden( self, filename ):
    
        with open( filename, 'w' ) as thefile:
            molden.header( self.mol, thefile )
            molden.orbital_coeff( self.mol, thefile, self.ao2loc )
            
    def loc_ortho( self ):
    
        ShouldBeI = np.dot( np.dot( self.ao2loc.T , self.mol.intor('cint1e_ovlp_sph') ) , self.ao2loc )
        return np.linalg.norm( ShouldBeI - np.eye( ShouldBeI.shape[0] ) )
        
    def const( self ):
    
        return self.mol.energy_nuc()
        
    def loc_oei( self ):
        
        return np.dot( np.dot( self.ao2loc.T , self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph') ) , self.ao2loc )
        
    def loc_rhf_fock( self ):
    
        Fockao  = self.JKao + self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
        Fockloc = np.dot( np.dot( self.ao2loc.T , Fockao ) , self.ao2loc )
        return Fockloc
        
    def loc_rhf_fock_bis( self, DMloc ):

        # DMloc is the RHF solution in the localized basis for a custom one-electron operator
        newDMao = np.dot( np.dot( self.ao2loc , DMloc ) , self.ao2loc.T )
        newJKao = scf.hf.get_veff( self.mol, newDMao, 0, 0, 1 )
        Fockao  = newJKao + self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
        Fockloc = np.dot( np.dot( self.ao2loc.T , Fockao ) , self.ao2loc )
        return Fockloc

    def loc_tei( self ):
    
        #return ao2mo.outcore.full_iofree( self.mol, self.ao2loc, compact=False ).reshape(Norb, Norb, Norb, Norb)
        return ao2mo.incore.full( self.eri, self.ao2loc, compact=False ).reshape(self.Norbs, self.Norbs, self.Norbs, self.Norbs)
        
    def dmet_oei( self, loc2dmet, numActive ):
    
        ao2dmet = np.dot( self.ao2loc , loc2dmet[:,:numActive] )
        OEIdmet = np.dot( np.dot( ao2dmet.T , self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph') ) , ao2dmet )
        return OEIdmet
        
    def dmet_fock( self, loc2dmet, numActive, core1RDM_loc ):
    
        DMao     = np.dot( np.dot( self.ao2loc , core1RDM_loc ) , self.ao2loc.T )
        JKao     = scf.hf.get_veff( self.mol, DMao, 0, 0, 1 ) #Last 3 numbers: dm_last, vhf_last, hermi
        FOCKao   = JKao + self.mol.intor('cint1e_kin_sph') + self.mol.intor('cint1e_nuc_sph')
        ao2dmet  = np.dot( self.ao2loc , loc2dmet[:,:numActive] )
        FOCKdmet = np.dot( np.dot( ao2dmet.T , FOCKao ) , ao2dmet )
        return FOCKdmet
        
    def dmet_tei( self, loc2dmet, numActive ):
    
        ao2dmet = np.dot( self.ao2loc , loc2dmet[:,:numActive] )
        #TEIdmet = ao2mo.outcore.full_iofree( self.mol, ao2dmet, compact=False ).reshape(numActive, numActive, numActive, numActive)
        TEIdmet = ao2mo.incore.full( self.eri, ao2dmet, compact=False ).reshape(numActive, numActive, numActive, numActive)
        return TEIdmet
        
        
