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
#importing required libraries
import localintegrals
import qcdmethelper
import numpy as np
from scipy import optimize
import time

class dmet:

    def __init__( self, theInts, impurityClusters, isTranslationInvariant, method='ED', SCmethod='LSTSQ', fitImpBath=True, use_constrained_opt=False ):
    
        if ( isTranslationInvariant == True ):
            assert( theInts.TI_OK == True )
        
        assert (( method == 'ED' ) or ( method == 'CC' ) or ( method == 'MP2' ))
        assert (( SCmethod == 'LSTSQ' ) or ( SCmethod == 'BFGS' ) or ( SCmethod == 'NONE' ))
        
        self.ints       = theInts
        self.Norb       = self.ints.Norbs
        self.impClust   = impurityClusters
        self.umat       = np.zeros([ self.Norb, self.Norb ], dtype=float)
        self.relaxation = 0.0
        
        self.NI_hack    = False
        self.method     = method
        self.doSCF      = False
        self.TransInv   = isTranslationInvariant
        self.SCmethod   = SCmethod
        self.CC_E_TYPE  = 'LAMBDA' #'CASCI'
        self.BATH_ORBS  = None
        self.fitImpBath = fitImpBath
        self.doDET      = False
        self.doDET_NO   = False
        self.NOrotation = None
        self.altcostfunc = use_constrained_opt

        self.minFunc    = None
        if self.altcostfunc:
            self.minFunc = 'FOCK_INIT'  # 'OEI'
            assert (self.fitImpBath == False)
            assert (self.doDET == False)
            assert (self.SCmethod == 'BFGS' or self.SCmethod == 'NONE')
        
        if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'CASCI' )):
            assert( len( self.impClust ) == 1 )
        
        if ( self.doDET == True ):
            # Cfr Bulik, PRB 89, 035140 (2014)
            self.fitImpBath = False
            if ( self.doDET_NO == True ):
                self.NOvecs = None
                self.NOdiag = None
        
        self.print_u   = True
        self.print_rdm = True
        
        allOne = self.testclusters()
        if ( allOne == False ): # One or more impurities which do not cover the entire system
            assert( self.TransInv == False ) # Make sure that you don't work translational invariant
            # Note on working with impurities which do no tile the entire system: they should be the first orbitals in the Hamiltonian!
        
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        self.imp_size = self.make_imp_size()
        self.mu_imp   = 0.0
        self.mask     = self.make_mask()
        self.helper   = qcdmethelper.qcdmethelper( self.ints, self.makelist_H1(), self.altcostfunc, self.minFunc )
        
        self.time_ed  = 0.0
        self.time_cf  = 0.0
        self.time_func= 0.0
        self.time_grad= 0.0
        
        np.set_printoptions(precision=3, linewidth=160)
        
    def testclusters( self ):
    
        quicktest = np.zeros([ self.Norb ], dtype=int)
        for item in self.impClust:
            quicktest += np.abs(item)
        assert( np.all( quicktest >= 0 ) )
        assert( np.all( quicktest <= 1 ) )
        allOne = np.all( quicktest == 1 )
        return allOne
            
    def make_imp_size( self ):
    
        thearray = []
        maxiter = len( self.impClust )
        if ( self.TransInv == True ):
            maxiter = 1
        for counter in range( maxiter ):
            impurityOrbs = np.abs(self.impClust[ counter ])
            numImpOrbs = np.sum( impurityOrbs )
            thearray.append( numImpOrbs )
        thearray = np.array( thearray )
        return thearray

    def makelist_H1( self ):
    
        theH1 = []
        if ( self.doDET == True ): # Do density embedding theory
            if ( self.TransInv == True ): # Translational invariance assumed
                localsize = self.imp_size[ 0 ]
                for row in range( localsize ):
                    H1 = np.zeros( [ self.Norb, self.Norb ], dtype=int )
                    for jumper in range( self.Norb / localsize ):
                        jumpsquare = localsize * jumper
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                    theH1.append( H1 )
            else: # NO translational invariance assumed
                jumpsquare = 0
                for localsize in self.imp_size:
                    for row in range( localsize ):
                        H1 = np.zeros( [ self.Norb, self.Norb ], dtype=int )
                        H1[ jumpsquare + row, jumpsquare + row ] = 1
                        theH1.append( H1 )
                    jumpsquare += localsize
        else: # Do density MATRIX embedding theory
            if ( self.TransInv == True ): # Translational invariance assumed
                localsize = self.imp_size[ 0 ]
                for row in range( localsize ):
                    for col in range( row, localsize ):
                        H1 = np.zeros( [ self.Norb, self.Norb ], dtype=int )
                        for jumper in range( self.Norb / localsize ):
                            jumpsquare = localsize * jumper
                            H1[ jumpsquare + row, jumpsquare + col ] = 1
                            H1[ jumpsquare + col, jumpsquare + row ] = 1
                        theH1.append( H1 )
            else: # NO translational invariance assumed
                jumpsquare = 0
                for localsize in self.imp_size:
                    for row in range( localsize ):
                        for col in range( row, localsize ):
                            H1 = np.zeros( [ self.Norb, self.Norb ], dtype=int )
                            H1[ jumpsquare + row, jumpsquare + col ] = 1
                            H1[ jumpsquare + col, jumpsquare + row ] = 1
                            theH1.append( H1 )
                    jumpsquare += localsize
        return theH1
        
    def make_mask( self ):
    
        themask = np.zeros( [ self.Norb, self.Norb ], dtype=bool )
        if ( self.doDET == True ): # Do density embedding theory
            jump = 0
            for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
                for row in range( localsize ):
                    themask[ jump + row, jump + row ] = True
                jump += localsize
        else: # Do density MATRIX embedding theory
            jump = 0
            for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
                for row in range( localsize ):
                    for col in range( row, localsize ):
                        themask[ jump + row, jump + col ] = True
                jump += localsize
        return themask
        
    def doexact( self, chempot_imp=0.0 ):
    
        OneRDM = self.helper.construct1RDM_loc( self.doSCF, self.umat )
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        if ( self.doDET == True ) and ( self.doDET_NO == True ):
            self.NOvecs = []
            self.NOdiag = []
        
        maxiter = len( self.impClust )
        if ( self.TransInv == True ):
            maxiter = 1
            
        remainingOrbs = np.ones( [ len( self.impClust[ 0 ] ) ], dtype=float )
        
        for counter in range( maxiter ):
        
            flag_rhf = np.sum(self.impClust[ counter ]) < 0
            impurityOrbs = np.abs(self.impClust[ counter ])
            numImpOrbs   = np.sum( impurityOrbs )
            if ( self.BATH_ORBS == None ):
                numBathOrbs = numImpOrbs
            else:
                numBathOrbs = self.BATH_ORBS[ counter ]
            numBathOrbs, loc2dmet, core1RDM_dmet = self.helper.constructbath( OneRDM, impurityOrbs, numBathOrbs )
            if ( self.BATH_ORBS == None ):
                core_cutoff = 0.01
            else:
                core_cutoff = 0.5
            for cnt in range(len(core1RDM_dmet)):
                if ( core1RDM_dmet[ cnt ] < core_cutoff ):
                    core1RDM_dmet[ cnt ] = 0.0
                elif ( core1RDM_dmet[ cnt ] > 2.0 - core_cutoff ):
                    core1RDM_dmet[ cnt ] = 2.0
                else:
                    print "Bad DMET bath orbital selection: trying to put a bath orbital with occupation", core1RDM_dmet[ cnt ], "into the environment :-(."
                    assert( 0 == 1 )

            Norb_in_imp  = numImpOrbs + numBathOrbs
            Nelec_in_imp = int(round(self.ints.Nelec - np.sum( core1RDM_dmet )))
            core1RDM_loc = np.dot( np.dot( loc2dmet, np.diag( core1RDM_dmet ) ), loc2dmet.T )
            
            self.dmetOrbs.append( loc2dmet[ :, :Norb_in_imp ] ) # Impurity and bath orbitals only
            assert( Norb_in_imp <= self.Norb )
            dmetOEI  = self.ints.dmet_oei(  loc2dmet, Norb_in_imp )
            dmetFOCK = self.ints.dmet_fock( loc2dmet, Norb_in_imp, core1RDM_loc )
            dmetTEI  = self.ints.dmet_tei(  loc2dmet, Norb_in_imp )
            
            if ( self.NI_hack == True ):
                dmetTEI[:,:,:,numImpOrbs:]=0.0
                dmetTEI[:,:,numImpOrbs:,:]=0.0
                dmetTEI[:,numImpOrbs:,:,:]=0.0
                dmetTEI[numImpOrbs:,:,:,:]=0.0
            
                umat_rotated = np.dot(np.dot(loc2dmet.T, self.umat), loc2dmet)
                umat_rotated[:numImpOrbs,:numImpOrbs]=0.0
                dmetOEI += umat_rotated[:Norb_in_imp,:Norb_in_imp]
                dmetFOCK = np.array( dmetOEI, copy=True )
            
            print "DMET::exact : Performing a (", Norb_in_imp, "orb,", Nelec_in_imp, "el ) DMET active space calculation."
            if ( flag_rhf ):
                import pyscf_rhf
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp/2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = pyscf_rhf.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, chempot_imp )
            elif ( self.method == 'ED' ):
                import chemps2
                IMP_energy, IMP_1RDM = chemps2.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, chempot_imp )
            elif ( self.method == 'CC' ):
                import pyscf_cc
                assert( Nelec_in_imp % 2 == 0 )
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp/2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = pyscf_cc.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, self.CC_E_TYPE, chempot_imp )
            elif ( self.method == 'MP2' ):
                import pyscf_mp2
                assert( Nelec_in_imp % 2 == 0 )
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, Norb_in_imp, Nelec_in_imp/2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = pyscf_mp2.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, Norb_in_imp, Nelec_in_imp, numImpOrbs, DMguessRHF, chempot_imp )
            self.energy += IMP_energy
            self.imp_1RDM.append( IMP_1RDM )
            if ( self.doDET == True ) and ( self.doDET_NO == True ):
                RDMeigenvals, RDMeigenvecs = np.linalg.eigh( IMP_1RDM[ :numImpOrbs, :numImpOrbs ] )
                self.NOvecs.append( RDMeigenvecs )
                self.NOdiag.append( RDMeigenvals )
                
            remainingOrbs -= impurityOrbs
        
        if ( self.doDET == True ) and ( self.doDET_NO == True ):
            self.NOrotation = self.constructNOrotation()
        
        Nelectrons = 0.0
        for counter in range( maxiter ):
            Nelectrons += np.trace( self.imp_1RDM[counter][ :self.imp_size[counter], :self.imp_size[counter] ] )
        if ( self.TransInv == True ):
            Nelectrons = Nelectrons * len( self.impClust )
            self.energy = self.energy * len( self.impClust )
            remainingOrbs[:] = 0
            
        # When an incomplete impurity tiling is used for the Hamiltonian, self.energy should be augmented with the remaining HF part
        if ( np.sum( remainingOrbs ) != 0 ):
        
            if ( self.CC_E_TYPE == 'CASCI' ):
                '''
                If CASCI is passed as CC energy type, the energy of the one and only full impurity Hamiltonian is returned.
                The one-electron integrals of this impurity Hamiltonian is the full Fock operator of the CORE orbitals!
                The constant part of the energy still needs to be added: sum_occ ( 2 * OEI[occ,occ] + JK[occ,occ] )
                                                                         = einsum( core1RDM_loc, OEI ) + 0.5 * einsum( core1RDM_loc, JK )
                                                                         = 0.5 * einsum( core1RDM_loc, OEI + FOCK )
                '''
                assert( maxiter == 1 )
                transfo = np.eye( self.Norb, dtype=float )
                totalOEI  = self.ints.dmet_oei(  transfo, self.Norb )
                totalFOCK = self.ints.dmet_fock( transfo, self.Norb, core1RDM_loc )
                self.energy += 0.5 * np.einsum( 'ij,ij->', core1RDM_loc, totalOEI + totalFOCK )
                Nelectrons = np.trace( self.imp_1RDM[ 0 ] ) + np.trace( core1RDM_loc ) # Because full active space is used to compute the energy
            else:
                #transfo = np.eye( self.Norb, dtype=float )
                #totalOEI  = self.ints.dmet_oei(  transfo, self.Norb )
                #totalFOCK = self.ints.dmet_fock( transfo, self.Norb, OneRDM )
                #self.energy += 0.5 * np.einsum( 'ij,ij->', OneRDM[remainingOrbs==1,:], \
                #         totalOEI[remainingOrbs==1,:] + totalFOCK[remainingOrbs==1,:] )
                #Nelectrons += np.trace( (OneRDM[remainingOrbs==1,:])[:,remainingOrbs==1] )

                assert (np.array_equal(self.ints.active, np.ones([self.ints.mol.nao_nr()], dtype=int)))

                from pyscf import scf
                from types import MethodType
                mol_ = self.ints.mol
                mf_  = scf.RHF(mol_)

                impOrbs = remainingOrbs==1
                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                hc  = -chempot_imp * np.dot(xorb[:,impOrbs], xorb[:,impOrbs].T)
                dm0 = np.dot(self.ints.ao2loc, np.dot(OneRDM, self.ints.ao2loc.T))

                def mf_hcore (self, mol=None):
                    if mol is None: mol = self.mol
                    return scf.hf.get_hcore(mol) + hc
                mf_.get_hcore = MethodType(mf_hcore, mf_)
                mf_.scf(dm0)
                assert (mf_.converged)

                rdm1 = mf_.make_rdm1()
                jk   = mf_.get_veff(dm=rdm1)

                xorb = np.dot(mf_.get_ovlp(), self.ints.ao2loc)
                rdm1 = np.dot(xorb.T, np.dot(rdm1, xorb))
                oei  = np.dot(self.ints.ao2loc.T, np.dot(mf_.get_hcore()-hc, self.ints.ao2loc))
                jk   = np.dot(self.ints.ao2loc.T, np.dot(jk, self.ints.ao2loc))

                ImpEnergy = \
                   + 0.50 * np.einsum('ji,ij->', rdm1[:,impOrbs], oei[impOrbs,:]) \
                   + 0.50 * np.einsum('ji,ij->', rdm1[impOrbs,:], oei[:,impOrbs]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[:,impOrbs], jk[impOrbs,:]) \
                   + 0.25 * np.einsum('ji,ij->', rdm1[impOrbs,:], jk[:,impOrbs])
                self.energy += ImpEnergy
                Nelectrons += np.trace(rdm1[np.ix_(impOrbs,impOrbs)])

            remainingOrbs[ remainingOrbs==1 ] -= 1
        assert( np.all( remainingOrbs == 0 ) )
            
        self.energy += self.ints.const()
        return Nelectrons
        
    def constructNOrotation( self ):
    
        myNOrotation = np.zeros( [ self.Norb, self.Norb ], dtype=float )
        jumpsquare = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            myNOrotation[ jumpsquare : jumpsquare + self.imp_size[ count ], jumpsquare : jumpsquare + self.imp_size[ count ] ] = self.NOvecs[ count ]
            jumpsquare += self.imp_size[ count ]
        for count in range( jumpsquare, self.Norb ):
            myNOrotation[ count, count ] = 1.0
        if ( self.TransInv == True ):
            size = self.imp_size[ 0 ]
            for it in range( 1, self.Norb / size ):
                myNOrotation[ it*size:(it+1)*size, it*size:(it+1)*size ] = myNOrotation[ 0:size, 0:size ]
        '''if True:
            assert ( np.linalg.norm( np.dot( myNOrotation.T, myNOrotation ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )
            assert ( np.linalg.norm( np.dot( myNOrotation, myNOrotation.T ) - np.eye( self.umat.shape[0] ) ) < 1e-10 )'''
        return myNOrotation
        
    def costfunction( self, newumatflat ):

        return np.linalg.norm( self.rdm_differences( newumatflat ) )**2

    def alt_costfunction( self, newumatflat ):

        newumatsquare_loc = self.flat2square( newumatflat )
        OneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )

        errors    = self.rdm_differences_bis( newumatflat )
        errors_sq = self.flat2square (errors)

        if self.minFunc == 'OEI' :
            e_fun = np.trace( np.dot(self.ints.loc_oei(), OneRDM_loc) )
        elif self.minFunc == 'FOCK_INIT' :
            e_fun = np.trace( np.dot(self.ints.loc_rhf_fock(), OneRDM_loc) )
        # e_cstr = np.sum( newumatflat * errors )    # not correct, but gives correct verify_gradient results
        e_cstr = np.sum( newumatsquare_loc * errors_sq )
        return -e_fun-e_cstr
        
    def costfunction_derivative( self, newumatflat ):
        
        errors = self.rdm_differences( newumatflat )
        error_derivs = self.rdm_differences_derivative( newumatflat )
        thegradient = np.zeros([ len( newumatflat ) ], dtype=float)
        for counter in range( len( newumatflat ) ):
            thegradient[ counter ] = 2 * np.sum( np.multiply( error_derivs[ : , counter ], errors ) )
        return thegradient

    def alt_costfunction_derivative( self, newumatflat ):
        
        errors = self.rdm_differences_bis( newumatflat )
        return -errors
    
    def rdm_differences( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare_loc = self.flat2square( newumatflat )
        OneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )
        
        thesize = 0
        for count in range(len(self.imp_size)):
            if ( self.doDET == True ): # Do density embedding theory: fit only impurity
                thesize += self.imp_size[ count ]
                assert ( self.fitImpBath == False )
            else: # Do density MATRIX embedding theory
                if ( self.fitImpBath == True ):
                    thesize += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    thesize += self.imp_size[ count ] * self.imp_size[ count ]
        errors = np.zeros( [ thesize ], dtype=float )
        
        jump = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            if ( self.fitImpBath == True ):
                mf_1RDM = np.dot( np.dot( self.dmetOrbs[ count ].T, OneRDM_loc ), self.dmetOrbs[ count ] )
                ed_1RDM = self.imp_1RDM[count]
            else:
                mf_1RDM = (OneRDM_loc[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
                ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            if ( self.doDET == True ): # Do density embedding theory
                if ( self.doDET_NO == True ): # Work in the NO basis
                    theerror = np.diag( np.dot( np.dot( self.NOvecs[ count ].T, mf_1RDM ), self.NOvecs[ count ] ) ) - self.NOdiag[ count ]
                else: # Work in the lattice basis
                    theerror = np.diag( mf_1RDM - ed_1RDM )
                errors[ jump : jump + len( theerror ) ] = theerror
                jump += len( theerror )
            else: # Do density MATRIX embedding theory
                theerror = mf_1RDM - ed_1RDM
                squaresize = theerror.shape[0] * theerror.shape[1]
                errors[ jump : jump + squaresize ] = np.reshape( theerror, squaresize, order='F' )
                jump += squaresize
        assert ( jump == thesize )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors
        
    def rdm_differences_bis( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare_loc = self.flat2square( newumatflat )
        OneRDM_loc = self.helper.construct1RDM_loc( self.doSCF, newumatsquare_loc )

        thesize = 0
        jump = 0
        for count in range(len(self.imp_size)):
            # thesize += self.imp_size[ count ] * self.imp_size[ count ]
            mask_t = self.mask[ np.ix_(range(jump,jump+self.imp_size[count]),range(jump,jump+self.imp_size[count])) ]
            thesize += np.count_nonzero( mask_t )
            jump += self.imp_size[count]
        errors = np.zeros( [ thesize ], dtype=float )
        
        jump = 0
        jumpc = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            mf_1RDM = (OneRDM_loc[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
            ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            theerror = mf_1RDM - ed_1RDM
            # squaresize = theerror.shape[0] * theerror.shape[1]
            # errors[ jump : jump + squaresize ] = np.reshape( theerror, squaresize, order='F' )
            mask_t = self.mask[ np.ix_(range(jumpc,jumpc+self.imp_size[count]),range(jumpc,jumpc+self.imp_size[count])) ]
            squaresize = np.count_nonzero( mask_t )
            errors[ jump : jump + squaresize ] = np.reshape( theerror[mask_t], squaresize, order='F' )
            jump  += squaresize
            jumpc += self.imp_size[count]
        assert ( jump == thesize )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors

    def rdm_differences_derivative( self, newumatflat ):
        
        start_grad = time.time()
        
        newumatsquare_loc = self.flat2square( newumatflat )
        RDMderivs_rot = self.helper.construct1RDM_response( self.doSCF, newumatsquare_loc, self.NOrotation )
        
        thesize = 0
        for count in range(len(self.imp_size)):
            if ( self.doDET == True ): # Do density embedding theory: fit only impurity
                thesize += self.imp_size[ count ]
                assert ( self.fitImpBath == False )
            else: # Do density MATRIX embedding theory
                if ( self.fitImpBath == True ):
                    thesize += self.dmetOrbs[count].shape[1] * self.dmetOrbs[count].shape[1]
                else:
                    thesize += self.imp_size[ count ] * self.imp_size[ count ]
        
        gradient = []
        for countgr in range( len( newumatflat ) ):          #range will be 0 to one less than newumatflat (stop value),with step value one
            error_deriv = np.zeros( [ thesize ], dtype=float )
            jump = 0
            jumpsquare = 0
            for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
                if ( self.fitImpBath == True ):
                    local_derivative = np.dot( np.dot( self.dmetOrbs[ count ].T, RDMderivs_rot[ countgr, :, : ] ), self.dmetOrbs[ count ] )
                else:
                    if ( self.doDET == True ) and ( self.doDET_NO == True ):
                        local_derivative = RDMderivs_rot[ countgr, jumpsquare : jumpsquare + self.imp_size[ count ],\
                                                                   jumpsquare : jumpsquare + self.imp_size[ count ] ]
                        jumpsquare += self.imp_size[ count ]
                    else:
                        local_derivative = ((RDMderivs_rot[ countgr, :, : ])[:,np.flatnonzero(self.impClust[count])])[np.flatnonzero(self.impClust[count]),:]
                if ( self.doDET == True ): # Do density embedding theory
                    local_derivative = np.diag( local_derivative )
                    error_deriv[ jump : jump + len( local_derivative ) ] = local_derivative
                    jump += len( local_derivative )
                else: # Do density MATRIX embedding theory
                    squaresize = local_derivative.shape[0] * local_derivative.shape[1]
                    error_deriv[ jump : jump + squaresize ] = np.reshape( local_derivative, squaresize, order='F' )
                    jump += squaresize
            assert ( jump == thesize )
            gradient.append( error_deriv )
        gradient = np.array( gradient ).T
        
        stop_grad = time.time()
        self.time_grad += ( stop_grad - start_grad )
        
        return gradient
        
    def verify_gradient( self, umatflat ):
    
        gradient = self.costfunction_derivative( umatflat )
        cost_reference = self.costfunction( umatflat )
        gradientbis = np.zeros( [ len( gradient ) ], dtype=float )
        stepsize = 1e-7
        for cnt in range( len( gradient ) ):
            umatbis = np.array( umatflat, copy=True )
            umatbis[cnt] += stepsize
            costbis = self.costfunction( umatbis )
            gradientbis[ cnt ] = ( costbis - cost_reference ) / stepsize
        print "   Norm( gradient difference ) =", np.linalg.norm( gradient - gradientbis )
        print "   Norm( gradient )            =", np.linalg.norm( gradient )
        
    def hessian_eigenvalues( self, umatflat ):
    
        stepsize = 1e-7
        gradient_reference = self.costfunction_derivative( umatflat )
        hessian = np.zeros( [ len( umatflat ), len( umatflat ) ], dtype=float )
        for cnt in range( len( umatflat ) ):
            gradient = umatflat.copy()
            gradient[ cnt ] += stepsize
            gradient = self.costfunction_derivative( gradient )
            hessian[ :, cnt ] = ( gradient - gradient_reference ) / stepsize
        hessian = 0.5 * ( hessian + hessian.T )
        eigvals, eigvecs = np.linalg.eigh( hessian )
        idx = eigvals.argsort()
        eigvals = eigvals[ idx ]
        eigvecs = eigvecs[ :, idx ]
        print "Hessian eigenvalues =", eigvals
        #print "Hessian 1st eigenvector =",eigvecs[:,0]
        #print "Hessian 2nd eigenvector =",eigvecs[:,1]
        
    def flat2square( self, umatflat ):
    
        umatsquare = np.zeros( [ self.Norb, self.Norb ], dtype=float )
        umatsquare[ self.mask ] = umatflat
        umatsquare = umatsquare.T
        umatsquare[ self.mask ] = umatflat
        if ( self.TransInv == True ):
            size = self.imp_size[ 0 ]
            for it in range( 1, self.Norb / size ):
                umatsquare[ it*size:(it+1)*size, it*size:(it+1)*size ] = umatsquare[ 0:size, 0:size ]
                
        '''if True:
            umatsquare_bis = np.zeros( [ self.Norb, self.Norb ], dtype=float )
            for cnt in range( len( umatflat ) ):
                umatsquare_bis += umatflat[ cnt ] * self.helper.list_H1[ cnt ]
            print "Verification flat2square = ", np.linalg.norm( umatsquare - umatsquare_bis )'''
        
        if ( self.NOrotation != None ):
            umatsquare = np.dot( np.dot( self.NOrotation, umatsquare ), self.NOrotation.T )
        return umatsquare
        
    def square2flat( self, umatsquare ):
    
        umatsquare_bis = np.array( umatsquare, copy=True )
        if ( self.NOrotation != None ):
            umatsquare_bis = np.dot( np.dot( self.NOrotation.T, umatsquare_bis ), self.NOrotation )
        umatflat = umatsquare_bis[ self.mask ]
        return umatflat
        
    def numeleccostfunction( self, chempot_imp ):
        
        Nelec_dmet   = self.doexact( chempot_imp )
        Nelec_target = self.ints.Nelec
        print "      (chemical potential , number of electrons) = (", chempot_imp, "," , Nelec_dmet ,")"
        return Nelec_dmet - Nelec_target

    def doselfconsistent( self ):
    
        iteration = 0
        u_diff = 1.0
        convergence_threshold = 1e-5
        print "RHF energy =", self.ints.fullEhf
        
        while ( u_diff > convergence_threshold ):
        
            iteration += 1
            print "DMET iteration", iteration
            umat_old = np.array( self.umat, copy=True )
            rdm_old = self.transform_ed_1rdm() # At the very first iteration, this matrix will be zero
            
            # Find the chemical potential for the correlated impurity problem
            start_ed = time.time()
            if (( self.method == 'CC' ) and ( self.CC_E_TYPE == 'CASCI' )):
                self.mu_imp = 0.0
                self.doexact( self.mu_imp )
            else:
                self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp )
                print "   Chemical potential =", self.mu_imp
            stop_ed = time.time()
            self.time_ed += ( stop_ed - start_ed )
            print "   Energy =", self.energy
            # self.verify_gradient( self.square2flat( self.umat ) ) # Only works for self.doSCF == False!!
            if ( self.SCmethod != 'NONE' and not(self.altcostfunc) ):
                self.hessian_eigenvalues( self.square2flat( self.umat ) )
            
            # Solve for the u-matrix
            start_cf = time.time()
            if ( self.altcostfunc and self.SCmethod == 'BFGS' ):
                result = optimize.minimize( self.alt_costfunction, self.square2flat( self.umat ), jac=self.alt_costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            elif ( self.SCmethod == 'LSTSQ' ):
                result = optimize.leastsq( self.rdm_differences, self.square2flat( self.umat ), Dfun=self.rdm_differences_derivative, factor=0.1 )
                self.umat = self.flat2square( result[ 0 ] )
            elif ( self.SCmethod == 'BFGS' ):
                result = optimize.minimize( self.costfunction, self.square2flat( self.umat ), jac=self.costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) ) # Remove arbitrary chemical potential shifts
            if ( self.altcostfunc ):
                print "   Cost function after convergence =", self.alt_costfunction( self.square2flat( self.umat ) )
            else:
                print "   Cost function after convergence =", self.costfunction( self.square2flat( self.umat ) )
            stop_cf = time.time()
            self.time_cf += ( stop_cf - start_cf )
            
            # Possibly print the u-matrix / 1-RDM
            if self.print_u:
                self.print_umat()
            if self.print_rdm:
                self.print_1rdm()
            
            # Get the error measure
            u_diff   = np.linalg.norm( umat_old - self.umat )
            rdm_diff = np.linalg.norm( rdm_old - self.transform_ed_1rdm() )
            self.umat = self.relaxation * umat_old + ( 1.0 - self.relaxation ) * self.umat
            print "   2-norm of difference old and new u-mat =", u_diff
            print "   2-norm of difference old and new 1-RDM =", rdm_diff
            print "******************************************************"
            
            if ( self.SCmethod == 'NONE' ):
                u_diff = 0.1 * convergence_threshold # Do only 1 iteration
        
        print "Time cf func =", self.time_func
        print "Time cf grad =", self.time_grad
        print "Time dmet ed =", self.time_ed
        print "Time dmet cf =", self.time_cf
        
        return self.energy
        
    def print_umat( self ):
    
        print "The u-matrix ="
        squarejumper = 0
        for localsize in self.imp_size: # self.imp_size has length 1 if self.TransInv
            print self.umat[ squarejumper:squarejumper+localsize , squarejumper:squarejumper+localsize ]
            squarejumper += localsize
    
    def print_1rdm( self ):
    
        print "The ED 1-RDM of the impurities ( + baths ) ="
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            print self.imp_1RDM[ count ]
            
    def transform_ed_1rdm( self ):
    
        result = np.zeros( [self.umat.shape[0], self.umat.shape[0]], dtype=float )
        squarejumper = 0
        for count in range( len( self.imp_1RDM ) ): # self.imp_size has length 1 if self.TransInv
            localsize = self.imp_size[ count ]
            result[ squarejumper:squarejumper+localsize , squarejumper:squarejumper+localsize ] = self.imp_1RDM[ count ][ :localsize , :localsize ]
            squarejumper += localsize
        return result
        
    def dump_bath_orbs( self, filename, impnumber=0 ):
        
        import qcdmet_paths
        from pyscf import tools
        from pyscf.tools import molden
        with open( filename, 'w' ) as thefile:
            molden.header( self.ints.mol, thefile )
            molden.orbital_coeff( self.ints.mol, thefile, np.dot( self.ints.ao2loc, self.dmetOrbs[impnumber] ) )
    
    def onedm_solution_rhf(self):
        return self.helper.construct1RDM_loc( self.doSCF, self.umat )
    
