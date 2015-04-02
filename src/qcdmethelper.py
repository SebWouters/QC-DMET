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
import rhf
import numpy as np

class qcdmethelper:

    def __init__( self, theLocalIntegrals ):
    
        self.locints = theLocalIntegrals
        
    def construct1RDM_loc( self, doFock, doSCF, umat ):
    
        assert( self.locints.Nelec % 2 == 0 )
        numPairs = self.locints.Nelec / 2
        
        if ( doFock == True ):
            OEI   = self.locints.loc_rhf_fock() + umat
            DMloc = self.construct1RDM_base( OEI, numPairs )
            if ( doSCF == True ):
                if ( self.locints.ERIinMEM == True ):
                    DMloc = rhf.solve_ERI( self.locints.loc_oei() + umat, self.locints.loc_tei(), DMloc, numPairs )
                else:
                    DMloc = rhf.solve_JK( self.locints.loc_oei() + umat, self.locints.mol, self.locints.ao2loc, DMloc, numPairs )
        else:
            OEI   = self.locints.loc_oei() + umat
            DMloc = self.construct1RDM_base( OEI, numPairs )
        return DMloc

    def construct1RDM_loc_response( self, doFock, doSCF, umat, list_H1 ):

        assert( self.locints.Nelec % 2 == 0 )
        numPairs = self.locints.Nelec / 2

        if ( doFock == True ):
            OEI = self.locints.loc_rhf_fock() + umat
            if ( doSCF == True ):
                DMloc = self.construct1RDM_base( OEI, numPairs )
                if ( self.locints.ERIinMEM == True ):
                    DMloc = rhf.solve_ERI( self.locints.loc_oei() + umat, self.locints.loc_tei(), DMloc, numPairs )
                else:
                    DMloc = rhf.solve_JK( self.locints.loc_oei() + umat, self.locints.mol, self.locints.ao2loc, DMloc, numPairs )
                OEI = self.locints.loc_rhf_fock_bis( DMloc ) + umat
        else:
            OEI = self.locints.loc_oei() + umat

        eigenvals, eigenvecs = np.linalg.eigh( OEI ) # Does not guarantee sorted eigenvectors!
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx] # Sorted eigenvalues and eigenvectors!
        OCCUPIED  = eigenvecs[ : , :numPairs ]
        VIRTUAL   = eigenvecs[ : , numPairs: ]
        
        RDMderivs = []
        for H1 in list_H1:
            WORK = np.dot( np.dot( VIRTUAL.T , H1 ) , OCCUPIED )
            for virt in range( WORK.shape[0] ):
                for occ in range( WORK.shape[1] ):
                    WORK[ virt, occ ] = - WORK[ virt, occ ] / ( eigenvals[ numPairs + virt ] - eigenvals[ occ ] )
            C_1 = np.dot( VIRTUAL, WORK )
            deriv = 2 * np.dot( OCCUPIED, C_1.T )
            deriv = deriv + deriv.T
            RDMderivs.append(deriv)
        return RDMderivs
        
    def construct1RDM_base( self, OEI, numPairs ):
    
        eigenvals, eigenvecs = np.linalg.eigh( OEI ) # Does not guarantee sorted eigenvectors!
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        OneDM = 2 * np.dot( eigenvecs[:,:numPairs] , eigenvecs[:,:numPairs].T )
        return OneDM
        
    def constructbath( self, OneDM, impurityOrbs ):
    
        embeddingOrbs = 1 - impurityOrbs
        embeddingOrbs = np.matrix( embeddingOrbs )
        if (embeddingOrbs.shape[0] > 1):
            embeddingOrbs = embeddingOrbs.T # Now certainly row-like matrix (shape = 1 x len(vector))
        isEmbedding = np.dot( embeddingOrbs.T , embeddingOrbs ) == 1
        numEmbedOrbs = np.sum( embeddingOrbs )
        embedding1RDM = np.reshape( OneDM[ isEmbedding ], ( numEmbedOrbs , numEmbedOrbs ) )

        numImpOrbs   = np.sum( impurityOrbs )
        numBathOrbs  = numImpOrbs # Value can be changed later for LR-QC-DMET
        numTotalOrbs = len( impurityOrbs )
        
        eigenvals, eigenvecs = np.linalg.eigh( embedding1RDM )
        idx = np.maximum( -eigenvals, eigenvals - 2.0 ).argsort() # Occupation numbers closest to 1 come first
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:,idx]
        pureEnvironEigVals = -eigenvals[numBathOrbs:]
        pureEnvironEigVecs = eigenvecs[:,numBathOrbs:]
        idx = pureEnvironEigVals.argsort()
        eigenvecs[:,numBathOrbs:] = pureEnvironEigVecs[:,idx]
        pureEnvironEigVals = -pureEnvironEigVals[idx]
        coreOccupations = np.hstack(( np.zeros([ numImpOrbs + numBathOrbs ]), pureEnvironEigVals ))
    
        for counter in range(0, numImpOrbs):
            eigenvecs = np.insert(eigenvecs, counter, 0.0, axis=1) #Stack columns with zeros in the beginning
        counter = 0
        for counter2 in range(0, numTotalOrbs):
            if ( impurityOrbs[counter2] ):
                eigenvecs = np.insert(eigenvecs, counter2, 0.0, axis=0) #Stack rows with zeros on locations of the impurity orbitals
                eigenvecs[counter2, counter] = 1.0
                counter += 1
        assert( counter == numImpOrbs )
    
        # Orthonormality is guaranteed due to (1) stacking with zeros and (2) orthonormality eigenvecs for symmetric matrix
        assert( np.linalg.norm( np.dot(eigenvecs.T, eigenvecs) - np.identity(numTotalOrbs) ) < 1e-12 )

        # eigenvecs[ : , 0:numImpOrbs ]                      = impurity orbitals
        # eigenvecs[ : , numImpOrbs:numImpOrbs+numBathOrbs ] = bath orbitals
        # eigenvecs[ : , numImpOrbs+numBathOrbs: ]           = pure environment orbitals in decreasing order of occupation number
        return ( eigenvecs, coreOccupations )
        
