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

class dmet:

    def __init__( self, theInts, impurityClusters, isTranslationInvariant, method='ED' ):
    
        if ( isTranslationInvariant == True ):
            assert( theInts.TI_OK == True )
        
        assert (( method == 'ED' ) or ( method == 'CC' ))
        
        self.ints     = theInts
        self.helper   = qcdmethelper.qcdmethelper( self.ints )
        self.Norb     = self.ints.Norbs
        self.impClust = impurityClusters
        self.umat     = np.zeros([ self.Norb, self.Norb ], dtype=float)
        
        self.method     = method
        self.doFock     = True # Do not dare to change this
        self.doSCF      = False
        self.TransInv   = isTranslationInvariant
        self.leastsq    = False # Upon setting to True all hell will break loose. Beware!
        self.fitImpBath = True
        
        self.print_u    = True
        self.print_rdm  = True
        
        '''self.densityfit = False
        if ( self.densityfit == True ):
            self.EDRDMeigvecs = []
            assert( self.fitImpBath == False )
            assert( self.leastsq == False )'''
        
        self.testclusters()
        
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        self.imp_size = self.make_imp_size()
        self.mu_imp   = 0.0
        self.list_H1  = self.makelist_H1()
        
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
        
    def doexact( self, chempot_imp=0.0 ):
    
        OneRDM = self.helper.construct1RDM_loc( self.doFock, self.doSCF, self.umat )
        self.energy   = 0.0
        self.imp_1RDM = []
        self.dmetOrbs = []
        '''if ( self.densityfit == True ):
            self.EDRDMeigvecs = []'''
        
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
                IMP_energy, IMP_1RDM = psi4cc.solve( 0.0, dmetOEI, dmetFOCK, dmetTEI, 2*numImpOrbs, Nelec_in_imp, numImpOrbs, chempot_imp )
            self.energy += IMP_energy
            self.imp_1RDM.append( IMP_1RDM )
            '''if ( self.densityfit == True ):
                eigvals, eigvecs = np.linalg.eigh( IMP_1RDM[:numImpOrbs,:numImpOrbs] )
                eigvecs = eigvecs[ :, eigvals.argsort() ]
                self.EDRDMeigvecs.append( eigvecs )'''
            
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
        thegradient = np.zeros([ len( self.list_H1 ) ], dtype=float)
        for counter in range( len( self.list_H1 ) ):
            thegradient[ counter ] = 2 * np.sum( np.multiply( error_derivs[ : , counter ], errors ) )
        return thegradient
        
    def rdm_differences( self, newumatflat ):
    
        newumatsquare = self.flat2square( newumatflat )
        OneRDM = self.helper.construct1RDM_loc( self.doFock, self.doSCF, newumatsquare )
        errors = []
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            if ( self.fitImpBath == True ):
                mf_1RDM = np.dot( np.dot( self.dmetOrbs[ count ].T, OneRDM ), self.dmetOrbs[ count ] )
                ed_1RDM = self.imp_1RDM[count]
            else:
                mf_1RDM = (OneRDM[:,self.impClust[count]==1])[self.impClust[count]==1,:]
                ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            theerror = mf_1RDM - ed_1RDM
            errors.append( theerror )
        errors = np.array( errors )
        errors = np.reshape( errors, errors.shape[0] * errors.shape[1] * errors.shape[2], order='F' )
        return errors
        
    def rdm_differences_derivative( self, newumatflat ):
        
        newumatsquare = self.flat2square( newumatflat )
        OneRDM = self.helper.construct1RDM_loc( self.doFock, self.doSCF, newumatsquare )
        RDMderivs = self.helper.construct1RDM_loc_response( self.doFock, self.doSCF, newumatsquare, self.list_H1 )
        
        gradient = []
        for countgr in range( len( self.list_H1 ) ):
            error_deriv = []
            for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
                if ( self.fitImpBath == True ):
                    local_derivative = np.dot( np.dot( self.dmetOrbs[ count ].T, RDMderivs[ countgr ] ), self.dmetOrbs[ count ] )
                else:
                    local_derivative = ((RDMderivs[ countgr ])[:,self.impClust[count]==1])[self.impClust[count]==1,:]
                error_deriv.append( local_derivative )
            error_deriv = np.array( error_deriv )
            error_deriv = np.reshape( error_deriv, error_deriv.shape[0] * error_deriv.shape[1] * error_deriv.shape[2], order='F' )
            gradient.append( error_deriv )
        gradient = np.array( gradient ).T
        return gradient
        
    def verify_gradient( self, umatflat ):
    
        self.doFock = not self.doFock #Invert to get a large gradient
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
        self.doFock = not self.doFock #Back to original case
        
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
        
    '''def density_costfunction( self, umatdiag ):
    
        umatsquare = self.diag2square( umatdiag )
        OneRDM = self.helper.construct1RDM_loc( self.doFock, self.doSCF, umatsquare )
        
        errors = []
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            mf_1RDM = (OneRDM[:,self.impClust[count]==1])[self.impClust[count]==1,:]
            ed_1RDM = self.imp_1RDM[count][:self.imp_size[count],:self.imp_size[count]]
            ed_dens = np.diag(np.dot(np.dot(self.EDRDMeigvecs[count].T, ed_1RDM), self.EDRDMeigvecs[count]))
            mf_dens = np.diag(np.dot(np.dot(self.EDRDMeigvecs[count].T, mf_1RDM), self.EDRDMeigvecs[count]))
            assert( np.linalg.norm( np.diag( np.diag( ed_dens ) ) - ed_dens ) < 1e-10 )
            errors.append( mf_dens - ed_dens )
        errors = np.array( errors )
        costfunction = np.linalg.norm( errors )**2
        return costfunction
        
    def diag2square( self, umatdiag ):
    
        umatsquare = np.zeros([ self.Norb, self.Norb ], dtype=float)
        jump = 0
        for count in range( len( self.imp_size ) ):
            localsize = self.imp_size[ count ]
            umatsquare[jump:jump+localsize,jump:jump+localsize] = np.dot(np.dot(self.EDRDMeigvecs[count], np.diag(umatdiag[jump:jump+localsize])), self.EDRDMeigvecs[count].T)
            jump += localsize
        if ( self.TransInv == True ):
            localsize = self.imp_size[ 0 ]
            for jumper in range( self.Norb / localsize ):
                jump = localsize*jumper
                umatsquare[jump:jump+localsize,jump:jump+localsize] = umatsquare[:localsize,:localsize]
        return umatsquare
        
    def square2diag( self, umatsquare ):
    
        totalsize = self.imp_size[0]
        for cnt in range(1, len(self.imp_size)):
            totalsize += self.imp_size[cnt]
        umatdiag = np.zeros([totalsize], dtype=float)
        jump = 0
        for cnt in range( len( self.imp_size ) ):
            localsize = self.imp_size[cnt]
            umatdiag[jump:jump+localsize] = np.diag(np.dot(np.dot(self.EDRDMeigvecs[cnt].T, umatsquare[jump:jump+localsize,jump:jump+localsize]), self.EDRDMeigvecs[cnt]))
            jump += localsize
        assert( jump == totalsize )
        return umatdiag'''
        
    def flat2square( self, umatflat ):
    
        '''if ( self.densityfit == True ):
            return self.diag2square( umatflat )'''
    
        umatsquare = np.zeros( [ self.Norb, self.Norb ], dtype=float )
        
        if ( self.TransInv == True ):
            localsize = self.imp_size[ 0 ]
            iterator = 0
            for row in range(localsize):
                for col in range(row, localsize):
                    for jumper in range( self.Norb / localsize ):
                        jumpsquare = localsize * jumper
                        umatsquare[ jumpsquare + row, jumpsquare + col ] = umatflat[ iterator ]
                        umatsquare[ jumpsquare + col, jumpsquare + row ] = umatsquare[ jumpsquare + row, jumpsquare + col ]
                    iterator += 1
        else:
            jumpsquare = 0
            jumpflat   = 0
            for count in range( len( self.imp_size ) ):
                localsize = self.imp_size[ count ]
                iterator = 0
                for row in range(localsize):
                    for col in range(row, localsize):
                        umatsquare[ jumpsquare + row, jumpsquare + col ] = umatflat[ jumpflat + iterator ]
                        umatsquare[ jumpsquare + col, jumpsquare + row ] = umatsquare[ jumpsquare + row, jumpsquare + col ]
                        iterator += 1
                jumpsquare += localsize
                jumpflat += iterator
        
        '''if True:
            umatsquare_bis = np.zeros( [ self.Norb, self.Norb ], dtype=float )
            for cnt in range( len( umatflat ) ):
                umatsquare_bis += umatflat[ cnt ] * self.list_H1[ cnt ]
            print "Verification flat2square = ", np.linalg.norm( umatsquare - umatsquare_bis )'''
        
        return umatsquare
            
    def square2flat( self, umatsquare ):
    
        '''if ( self.densityfit == True ):
            return self.square2diag( umatsquare )'''
    
        umatflat = []
        jumpsquare = 0
        for count in range( len( self.imp_size ) ): # self.imp_size has length 1 if self.TransInv
            localsize = self.imp_size[ count ]
            for row in range(localsize):
                for col in range(row, localsize):
                    umatflat.append( umatsquare[ jumpsquare + row, jumpsquare + col ] )
            jumpsquare += localsize
        umatflat = np.array( umatflat )
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
        
        # Make sure that the initial bath orbitals are calculated based on the Fock matrix
        #    If self.doFock : self.umat = 0
        #    If not self.doFock : self.umat = FOCK - OEI
        #    JK matrix elements are small between different impurities as they are proportional to orbital overlaps between different impurities
        if ( np.linalg.norm( self.umat ) == 0.0 ) and ( not self.doFock ):
            self.umat = self.flat2square( self.square2flat( self.ints.loc_rhf_fock() - self.ints.loc_oei() ) )
        
        while ( u_diff > convergence_threshold ):
        
            iteration += 1
            print "DMET iteration", iteration
            umat_old = np.array( self.umat, copy=True )
            rdm_old = self.transform_ed_1rdm()
            
            # Find the chemical potential for the correlated impurity problem
            #self.mu_imp = optimize.brentq( self.numeleccostfunction, self.mu_imp - 10*u_diff , self.mu_imp + 10*u_diff )
            self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp )
            print "   Chemical potential =", self.mu_imp
            print "   Energy =", self.energy
            #self.hessian_eigenvalues( self.square2flat( self.umat ) )
            
            # Solve for the u-matrix
            umatflat = self.square2flat( self.umat )
            if ( self.leastsq == True ):
                result = optimize.leastsq( self.rdm_differences, umatflat, Dfun=self.rdm_differences_derivative )
                self.umat = self.flat2square( result[ 0 ] )
            else:
                '''if ( self.densityfit == True ):
                    result = optimize.minimize( self.density_costfunction, umatflat, options={'disp': False} )
                else:
                    result = optimize.minimize( self.costfunction, umatflat, jac=self.costfunction_derivative, options={'disp': False} )'''
                result = optimize.minimize( self.costfunction, umatflat, jac=self.costfunction_derivative, options={'disp': False} )
                self.umat = self.flat2square( result.x )
            self.umat = self.umat - np.eye( self.umat.shape[ 0 ] ) * np.average( np.diag( self.umat ) ) # Remove arbitrary chemical potential shifts
            '''if ( self.densityfit == True ):
                print "   Cost function after convergence =", self.density_costfunction( self.square2flat( self.umat ) )
            else:
                print "   Cost function after convergence =", self.costfunction( self.square2flat( self.umat ) )'''
            print "   Cost function after convergence =", self.costfunction( self.square2flat( self.umat ) )
            #self.verify_gradient( self.square2flat( self.umat ) )
            #self.hessian_eigenvalues( self.square2flat( self.umat ) )
            
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
            
        #gradient = self.energy_gradient( self.square2flat(self.umat) )
        #print "Energy gradient w.r.t. u-matrix =", gradient
        #self.energy_hessian()
        #result = optimize.minimize( self.energy_costfunction, self.square2flat( self.umat ), options={'disp': False} )
        return self.energy
        
    def energy_hessian( self ):
    
        stepsize = 1e-6
        umat_flat = self.square2flat( self.umat )
        hessian = np.zeros([ len(umat_flat), len(umat_flat) ], dtype=float)
        for cnt in range( len(umat_flat) ):
            umat_flat = self.square2flat( self.umat )
            umat_flat[ cnt ] += stepsize
            hessian[ :, cnt ] = self.energy_gradient( umat_flat )
        hessian = 0.5 * ( hessian + hessian.T )
        eigvals, eigvecs = np.linalg.eigh( hessian )
        idx = eigvals.argsort()
        eigvals = eigvals[ idx ]
        eigvecs = eigvecs[ :, idx ]
        print "Energy hessian eigenvalues =", eigvals
        
    def energy_gradient( self, umat_flat ):
    
        energy_orig = self.energy_costfunction( umat_flat )
        stepsize = 1e-6
        gradient = np.zeros( [ len(umat_flat) ], dtype=float )
        for cnt in range( len( umat_flat ) ):
            umat_flat_bis = np.array( umat_flat, copy=True )
            umat_flat_bis[cnt] += stepsize
            energy_new = self.energy_costfunction( self, umat_flat_bis )
            gradient[cnt] = ( energy_new - energy_orig ) / stepsize
        return gradient
        
    def energy_costfunction( self, umat_flat ):
    
        umat_orig = np.array( self.umat, copy=True )
        self.umat = self.flat2square( umat_flat )
        self.mu_imp = optimize.newton( self.numeleccostfunction, self.mu_imp )
        self.umat = np.array( umat_orig, copy=True )
        print "dmet::energy_costfunction : Current value of the energy =", self.energy
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
        
