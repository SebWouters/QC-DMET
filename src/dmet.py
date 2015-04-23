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

import localintegrals
import qcdmethelper
import numpy as np
from scipy import optimize
import time

class dmet:

    def __init__( self, theInts, impurityClusters, isTranslationInvariant, method='ED' ):
    
        if ( isTranslationInvariant == True ):
            assert( theInts.TI_OK == True )
        
        assert (( method == 'ED' ) or ( method == 'CC' ))
        
        self.ints     = theInts
        self.Norb     = self.ints.Norbs
        self.impClust = impurityClusters
        self.umat     = np.zeros([ self.Norb, self.Norb ], dtype=float)
        
        self.method     = method
        self.doSCF      = False
        self.TransInv   = isTranslationInvariant
        self.leastsq    = True
        self.fitImpBath = True # Fitting the impurity plus bath is **way** more stable
        
        self.print_u    = True
        self.print_rdm  = True
        
        self.testclusters()
        
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        self.imp_size = self.make_imp_size()
        self.mu_imp   = 0.0
        self.mask     = self.make_mask()
        self.helper   = qcdmethelper.qcdmethelper( self.ints, self.makelist_H1() )
        
        self.time_ed  = 0.0
        self.time_cf  = 0.0
        self.time_func= 0.0
        self.time_grad= 0.0
        
        np.set_printoptions(precision=3, linewidth=200)
        
    def testclusters( self ):
    
        quicktest = np.zeros([ self.Norb ], dtype=int)
        for item in self.impClust:
            quicktest += item
        for numcounts in quicktest:
            assert( numcounts == 1 )
            
    def make_imp_size( self ):
    
        thearray = []
        maxiter = len( self.impClust )
        if ( self.TransInv == True ):
            maxiter = 1
        for counter in range( maxiter ):
            impurityOrbs = self.impClust[ counter ]
            numImpOrbs = np.sum( impurityOrbs )
            thearray.append( numImpOrbs )
        thearray = np.array( thearray )
        return thearray

    def makelist_H1( self ):
    
        theH1 = []
        if ( self.TransInv == True ):
            localsize = self.imp_size[ 0 ]
            for row in range( localsize ):
                for col in range( row, localsize ):
                    H1 = np.zeros( [ self.Norb, self.Norb ], dtype=int )
                    for jumper in range( self.Norb / localsize ):
                        jumpsquare = localsize * jumper
                        H1[ jumpsquare + row, jumpsquare + col ] = 1
                        H1[ jumpsquare + col, jumpsquare + row ] = 1
                    theH1.append( H1 )
        else:
            jumpsquare = 0
            for count in range( len( self.imp_size ) ):
                localsize = self.imp_size[ count ]
                for row in range( localsize ):
                    for col in range( row, localsize ):
                        H1 = np.zeros( [ self.Norb, self.Norb ], dtype=int )
                        H1[ jumpsquare + row, jumpsquare + col ] = 1
                        H1[ jumpsquare + col, jumpsquare + row ] = 1
                        theH1.append( H1 )
                jumpsquare += localsize
        return theH1
        
    def make_mask( self ):
    
        themask = np.zeros([self.Norb,self.Norb], dtype=bool)
        jump = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            localsize = self.imp_size[ count ]
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
        
        maxiter = len( self.impClust )
        if ( self.TransInv == True ):
            maxiter = 1
        
        for counter in range( maxiter ):
        
            impurityOrbs = self.impClust[ counter ]
            loc2dmet, core1RDM_dmet = self.helper.constructbath( OneRDM, impurityOrbs )
            core1RDM_loc = np.dot( np.dot( loc2dmet, np.diag( core1RDM_dmet ) ), loc2dmet.T )
            
            numImpOrbs = np.sum( impurityOrbs )
            self.dmetOrbs.append( loc2dmet[ :, :2*numImpOrbs ] ) # Impurity and bath orbitals only
            assert( 2*numImpOrbs <= self.Norb )
            dmetOEI  = self.ints.dmet_oei(  loc2dmet, 2*numImpOrbs )
            dmetFOCK = self.ints.dmet_fock( loc2dmet, 2*numImpOrbs, core1RDM_loc )
            dmetTEI  = self.ints.dmet_tei(  loc2dmet, 2*numImpOrbs )
            
            Nelec_in_imp = 2*numImpOrbs
            if ( Nelec_in_imp > self.ints.Nelec ):
                Nelec_in_imp = self.ints.Nelec
            if ( self.method == 'ED' ):
                import chemps2
                IMP_energy, IMP_1RDM = chemps2.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, 2*numImpOrbs, Nelec_in_imp, numImpOrbs, chempot_imp )
            if ( self.method == 'CC' ):
                import psi4cc
                assert( Nelec_in_imp % 2 == 0 )
                DMguessRHF = self.ints.dmet_init_guess_rhf( loc2dmet, 2*numImpOrbs, Nelec_in_imp/2, numImpOrbs, chempot_imp )
                IMP_energy, IMP_1RDM = psi4cc.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, 2*numImpOrbs, Nelec_in_imp, numImpOrbs, DMguessRHF, chempot_imp )
            self.energy += IMP_energy
            self.imp_1RDM.append( IMP_1RDM )
            
        Nelectrons = 0.0
        for counter in range( maxiter ):
            IMP_1RDM = self.imp_1RDM[ counter ]
            IMP_Size = self.imp_size[ counter ]
            Nelectrons += np.trace( IMP_1RDM[ :IMP_Size, :IMP_Size ] )
        if ( self.TransInv == True ):
            Nelectrons = Nelectrons * len( self.impClust )
            self.energy = self.energy * len( self.impClust )
        self.energy += self.ints.const()
        return Nelectrons
        
    def costfunction( self, newumatflat ):

        return np.linalg.norm( self.rdm_differences( newumatflat ) )**2
        
    def costfunction_derivative( self, newumatflat ):
        
        errors = self.rdm_differences( newumatflat )
        error_derivs = self.rdm_differences_derivative( newumatflat )
        thegradient = np.zeros([ len( newumatflat ) ], dtype=float)
        for counter in range( len( newumatflat ) ):
            thegradient[ counter ] = 2 * np.sum( np.multiply( error_derivs[ : , counter ], errors ) )
        return thegradient
    
    def rdm_differences( self, newumatflat ):
    
        start_func = time.time()
    
        newumatsquare = self.flat2square( newumatflat )
        OneRDM = self.helper.construct1RDM_loc( self.doSCF, newumatsquare )
        
        thesize = 0
        for item in self.imp_size:
            if ( self.fitImpBath == True ):
                thesize += 4 * item * item
            else:
                thesize += item * item
        errors = np.zeros( [ thesize ], dtype=float )
        
        jump = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            if ( self.fitImpBath == True ):
                mf_1RDM = np.dot( np.dot( self.dmetOrbs[ count ].T, OneRDM ), self.dmetOrbs[ count ] )
                ed_1RDM = self.imp_1RDM[count]
            else:
                mf_1RDM = (OneRDM[:,self.impClust[count]==1])[self.impClust[count]==1,:]
                ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            theerror = mf_1RDM - ed_1RDM
            squaresize = theerror.shape[0] * theerror.shape[1]
            errors[ jump : jump + squaresize ] = np.reshape( theerror, squaresize, order='F' )
            jump += squaresize
        assert ( jump == thesize )
        
        stop_func = time.time()
        self.time_func += ( stop_func - start_func )
        
        return errors
    
    def rdm_differences_derivative( self, newumatflat ):
        
        start_grad = time.time()
        
        newumatsquare = self.flat2square( newumatflat )
        OneRDM, RDMderivs = self.helper.construct1RDM_loc_response_c( self.doSCF, newumatsquare )
        
        thesize = 0
        for item in self.imp_size:
            if ( self.fitImpBath == True ):
                thesize += 4 * item * item
            else:
                thesize += item * item
        
        gradient = []
        for countgr in range( len( newumatflat ) ):
            error_deriv = np.zeros( [ thesize ], dtype=float )
            jump = 0
            for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
                if ( self.fitImpBath == True ):
                    local_derivative = np.dot( np.dot( self.dmetOrbs[ count ].T, RDMderivs[ countgr, :, : ] ), self.dmetOrbs[ count ] )
                else:
                    local_derivative = ((RDMderivs[ countgr, :, : ])[:,self.impClust[count]==1])[self.impClust[count]==1,:]
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
        gradientbis = np.zeros( [len(gradient)], dtype=float )
        stepsize = 1e-7
        for cnt in range(len(gradient)):
            umatbis = np.array( umatflat, copy=True )
            umatbis[cnt] += stepsize
            costbis = self.costfunction( umatbis )
            gradientbis[ cnt ] = ( costbis - cost_reference ) / stepsize
        print "   Norm( gradient difference ) =", np.linalg.norm( gradient - gradientbis )
        print "   Norm( gradient )            =", np.linalg.norm( gradient )
        
    def hessian_eigenvalues( self, umatflat ):
    
        stepsize = 1e-7
        gradient_reference = self.costfunction_derivative( umatflat )
        hessian = np.zeros( [ len(umatflat), len(umatflat) ], dtype=float )
        for cnt in range(len(umatflat)):
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
        print "Hessian 1st eigenvector =",eigvecs[:,0]
        print "Hessian 2nd eigenvector =",eigvecs[:,1]
        
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
        
        return umatsquare
        
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
            rdm_old = self.transform_ed_1rdm()
            
            # Find the chemical potential for the correlated impurity problem
            start_ed = time.time()
            self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp )
            stop_ed = time.time()
            self.time_ed += ( stop_ed - start_ed )
            print "   Chemical potential =", self.mu_imp
            print "   Energy =", self.energy
            #self.verify_gradient( self.umat[ self.mask ] ) # Only works for self.doSCF == False!!
            #self.hessian_eigenvalues( self.umat[ self.mask ] )
            
            # Solve for the u-matrix
            start_cf = time.time()
            if ( self.leastsq == True ):
                result = optimize.leastsq( self.rdm_differences, self.umat[ self.mask ], Dfun=self.rdm_differences_derivative, factor=0.1 )
                self.umat = self.flat2square( result[ 0 ] )
            else:
                result = optimize.minimize( self.costfunction, self.umat[ self.mask ], jac=self.costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) ) # Remove arbitrary chemical potential shifts
            print "   Cost function after convergence =", self.costfunction( self.umat[ self.mask ] )
            #self.hessian_eigenvalues( self.umat[ self.mask ] )
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
            print "   2-norm of difference old and new u-mat =", u_diff
            print "   2-norm of difference old and new 1-RDM =", rdm_diff
            print "******************************************************"
            
            #u_diff = 0.1 * convergence_threshold # Do only 1 iteration
        
        print "Time cf func =", self.time_func
        print "Time cf grad =", self.time_grad
        print "Time dmet ed =", self.time_ed
        print "Time dmet cf =", self.time_cf
        
        return self.energy
        
    def print_umat( self ):
    
        print "The u-matrix ="
        squarejumper = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            localsize = self.imp_size[ count ]
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
        
