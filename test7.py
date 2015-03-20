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
sys.path.append('src')
import localintegrals, dmet
from pyscf import gto, scf, symm
import numpy as np

mol = gto.Mole() # Buckyball optimized with psi4 B3LYP/cc-pVDZ
mol.atom = '''
    C            0.000000000000    -0.698863692631     3.485659405234
    C           -0.000000000000     0.698863692631     3.485659405234
    C           -1.177413455163    -1.426586745678     3.035930309273
    C           -2.308221759410    -0.727692905561     2.604000392388
    C           -0.727692843917    -2.604000227480     2.308221628935
    C           -1.426586686597    -3.035930312767     1.177413473440
    C            0.727692843917    -2.604000227480     2.308221628935
    C            1.426586686597    -3.035930312767     1.177413473440
    C            1.177413455163    -1.426586745678     3.035930309273
    C            2.308221759410    -0.727692905561     2.604000392388
    C            1.426586686597     3.035930312767     1.177413473440
    C            0.727692843917     2.604000227480     2.308221628935
    C            0.698863670302     3.485659413332     0.000000000000
    C           -0.698863670302     3.485659413332     0.000000000000
    C            2.604000385623     2.308221792277    -0.727692910676
    C            3.035930522186     1.177413550960    -1.426586813835
    C            2.604000385623     2.308221792277     0.727692910676
    C            3.035930522186     1.177413550960     1.426586813835
    C            1.426586686597     3.035930312767    -1.177413473440
    C            0.727692843917     2.604000227480    -2.308221628935
    C           -0.727692843917     2.604000227480     2.308221628935
    C           -1.426586686597     3.035930312767     1.177413473440
    C           -1.177413455163     1.426586745678     3.035930309273
    C           -2.308221759410     0.727692905561     2.604000392388
    C            1.177413455163     1.426586745678     3.035930309273
    C            2.308221759410     0.727692905561     2.604000392388
    C           -2.604000385623     2.308221792277     0.727692910676
    C           -3.035930522186     1.177413550960     1.426586813835
    C           -2.604000385623     2.308221792277    -0.727692910676
    C           -3.035930522186     1.177413550960    -1.426586813835
    C           -1.426586686597     3.035930312767    -1.177413473440
    C           -0.727692843917     2.604000227480    -2.308221628935
    C           -3.485659767723    -0.000000000000     0.698863749657
    C           -3.485659767723    -0.000000000000    -0.698863749657
    C           -3.035930522186    -1.177413550960     1.426586813835
    C           -2.604000385623    -2.308221792277     0.727692910676
    C           -3.035930522186    -1.177413550960    -1.426586813835
    C           -2.604000385623    -2.308221792277    -0.727692910676
    C           -2.308221759410    -0.727692905561    -2.604000392388
    C           -1.177413455163    -1.426586745678    -3.035930309273
    C           -2.308221759410     0.727692905561    -2.604000392388
    C           -1.177413455163     1.426586745678    -3.035930309273
    C           -1.426586686597    -3.035930312767    -1.177413473440
    C           -0.727692843917    -2.604000227480    -2.308221628935
    C           -0.698863670302    -3.485659413332     0.000000000000
    C            0.698863670302    -3.485659413332     0.000000000000
    C            0.727692843917    -2.604000227480    -2.308221628935
    C            1.426586686597    -3.035930312767    -1.177413473440
    C            1.177413455163    -1.426586745678    -3.035930309273
    C            2.308221759410    -0.727692905561    -2.604000392388
    C            0.000000000000    -0.698863692631    -3.485659405234
    C           -0.000000000000     0.698863692631    -3.485659405234
    C            2.604000385623    -2.308221792277    -0.727692910676
    C            3.035930522186    -1.177413550960    -1.426586813835
    C            2.604000385623    -2.308221792277     0.727692910676
    C            3.035930522186    -1.177413550960     1.426586813835
    C            3.485659767723     0.000000000000    -0.698863749657
    C            3.485659767723     0.000000000000     0.698863749657
    C            2.308221759410     0.727692905561    -2.604000392388
    C            1.177413455163     1.426586745678    -3.035930309273
  '''
mol.basis = '6-31g'
mol.symmetry = 0
mol.charge = 0
mol.spin = 0 #2*S; multiplicity-1
mol.build()

mf = scf.RHF( mol )
mf.verbose = 4
mf.scf()

'''from pyscf.tools import molden, localizer
with open( 'buckyball-mo.molden', 'w' ) as thefile:
   molden.header( mol, thefile )
   molden.orbital_coeff( mol, thefile, mf.mo_coeff )
exit(123)'''

active = np.hstack((np.array([ 121, 131, 132, 133, 145, 146, 147, 148, 149 ]), range( 160, 211 ))) - 1

myInts = localintegrals.localintegrals( mf, active, 'boys' )
for atom in range( mol.natm ):
    # co = [ cos(phi)sin(theta), sin(phi)sin(theta), cos(theta) ] = unit vector pointing from origin to carbon atom
    # p_r = co_x * p_x + co_y * p_y + co_z * p_z
    co = mol.atom_coord( atom ) / np.linalg.norm( mol.atom_coord( atom ) )
    coeff_pxyz = myInts.ao2loc[ 9*atom+3 : 9*atom+6, atom ] # 9 basisfunctions per atom in 6-31G
    coeff_pr = np.einsum( 'i,i->', coeff_pxyz, co ) # are all 0.5526 * ( \pm 1 )
    if ( coeff_pr < 0.0 ):
        myInts.ao2loc[ :, atom ] = - myInts.ao2loc[ :, atom ]
myInts.molden( 'buckyball.molden' )
#myInts.exact_reference()

nAtomPerImp = 2
impurityClusters = []

if ( nAtomPerImp == 1 ):
    for cluster in range(len(active)):
        impurities = np.zeros( [ len(active) ], dtype=int )
        impurities[ cluster ] = 1
        impurityClusters.append( impurities )
    myInts.TI_OK = True # Overwrite that translational invariance is OK
if ( nAtomPerImp == 2 ):
    lowbound = 2.641 - 0.01
    upbound  = 2.641 + 0.01
    distances = np.zeros( [ mol.natm, mol.natm ], dtype=float )
    for atom1 in range( mol.natm ):
        for atom2 in range( atom1+1, mol.natm ):
            thisdistance = np.linalg.norm( mol.atom_coord( atom1 ) - mol.atom_coord( atom2 ) )
            distances[ atom1, atom2 ] = thisdistance
            distances[ atom2, atom1 ] = thisdistance
    notused = np.ones( [ mol.natm ], dtype=int )
    for atom in range( mol.natm ):
        if ( notused[ atom ] == 1 ):
            friend = np.multiply( ( distances[ atom, : ] > lowbound ), ( distances[ atom, : ] < upbound ) ).nonzero()[0][0]
            impurities = np.zeros( [ len(active) ], dtype=int )
            impurities[ atom   ] = 1
            impurities[ friend ] = 1
            impurityClusters.append( impurities )
            notused[ atom   ] = 0
            notused[ friend ] = 0
    myInts.TI_OK = True # Overwrite that translational invariance is OK
if ( nAtomPerImp == 5 ):
    clusters = np.zeros( [ 12, nAtomPerImp ], dtype=int )
    clusters[0, :] = np.array([ 16, 57, 54, 50, 59 ]) - 1 #OK
    clusters[1, :] = np.array([ 52, 42, 32, 20, 60 ]) - 1 #OK
    clusters[2, :] = np.array([ 15, 19, 13, 11, 17 ]) - 1 #OK
    clusters[3, :] = np.array([ 56, 58, 18, 26, 10 ]) - 1 #OK
    clusters[4, :] = np.array([ 8,  46, 48, 53, 55 ]) - 1 #OK
    clusters[5, :] = np.array([ 1,  3,  5,  7,  9  ]) - 1 #OK
    clusters[6, :] = np.array([ 25, 12, 21, 23, 2  ]) - 1 #OK
    clusters[7, :] = np.array([ 31, 29, 27, 22, 14 ]) - 1 #OK
    clusters[8, :] = np.array([ 4,  24, 28, 33, 35 ]) - 1 #OK
    clusters[9, :] = np.array([ 41, 39, 37, 34, 30 ]) - 1 #OK
    clusters[10,:] = np.array([ 6,  36, 38, 43, 45 ]) - 1 #OK
    clusters[11,:] = np.array([ 51, 49, 47, 44, 40 ]) - 1 #OK
    if ( True ):
        ao2loc_old = myInts.ao2loc.copy()
        for cnt in range( 12 ):
            myInts.ao2loc[ :, cnt*nAtomPerImp : (cnt+1)*nAtomPerImp ] = ao2loc_old[ :, clusters[cnt, :] ]
        myInts.molden( 'buckyball.molden' )
    for cnt in range( 12 ):
        impurities = np.zeros( [ len( active ) ], dtype=int )
        impurities[ cnt*nAtomPerImp : (cnt+1)*nAtomPerImp ] = 1
        impurityClusters.append( impurities )
    myInts.TI_OK = True # Overwrite that translational invariance is OK

isTranslationInvariant = True
theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant )
theDMET.doselfconsistent()


