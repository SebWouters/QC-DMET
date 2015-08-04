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
sys.path.append('../src')
import localintegrals_hubbard, dmet
import numpy as np

HubbardU   = 1.0
Norbs      = 240
imp_size   = 2

assert ( Norbs % imp_size == 0 )

fillings = []
energies = []

for Nelectrons in range( 12, 241, 12 ):

   hopping  = np.zeros( [Norbs, Norbs], dtype=float )
   for orb in range(Norbs-1):
       hopping[ orb, orb+1 ] = -1.0
       hopping[ orb+1, orb ] = -1.0
   hopping[ 0, Norbs-1 ] = 1.0 # anti-PBC
   hopping[ Norbs-1, 0 ] = 1.0 # anti-PBC

   myInts = localintegrals_hubbard.localintegrals_hubbard( hopping, HubbardU, Nelectrons )

   impurityClusters = []
   for cluster in range( Norbs / imp_size ):
       impurities = np.zeros( [ myInts.Norbs ], dtype=int )
       for orb in range( cluster*imp_size, (cluster+1)*imp_size ):
           impurities[ orb ] = 1
       impurityClusters.append( impurities )

   totalcount = np.zeros( [ myInts.Norbs ], dtype=int )
   for item in impurityClusters:
       totalcount += item
   assert ( np.linalg.norm( totalcount - np.ones( [ myInts.Norbs ], dtype=float ) ) < 1e-12 )

   isTranslationInvariant = True
   method = 'ED'
   SCmethod = 'LSTSQ' # 'LSTSQ'
   doSCF = False
   theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod, doSCF )
   #oldUMAT = 0.33 * ( 2 * np.random.rand( Norbs, Norbs ) - 1 )
   #if ( oldUMAT != None ):
   #    theDMET.umat = theDMET.flat2square( theDMET.square2flat( oldUMAT ) )
   theEnergy = theDMET.doselfconsistent()
   
   fillings.append( (1.0 * Nelectrons) / Norbs )
   energies.append( theEnergy / Norbs )

np.set_printoptions(precision=8, linewidth=160)
print "For U =", HubbardU,"and Norbs =", Norbs
print "Fillings ="
print np.array( fillings )
print "E / site ="
print np.array( energies )


