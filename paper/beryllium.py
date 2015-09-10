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
import localintegrals, dmet, ringhelper
from pyscf import gto, scf, future
from pyscf.future import cc
from pyscf.future.cc import ccsd
import numpy as np

###  Disclaimer: run one of the three cases for root following
casenumber = 3 # 1, 2 or 3

if ( casenumber == 1 ):
    thecases = np.arange( 3.6, 2.88, -0.1 )
if ( casenumber == 2 ):
    thecases = np.arange( 2.4, 2.92, +0.1 )
if ( casenumber == 3 ):
    thecases = np.arange( 2.4, 1.78, -0.1 )
    
print "Bond lengths (Angstrom) =", thecases

DMguess = None
for bl in thecases:

    nat = 12
    mol = gto.Mole()
    mol.atom = []
    r = 0.5 * bl / np.sin(np.pi/nat)
    for i in range(nat):
        theta = i * (2*np.pi/nat)
        mol.atom.append(('Be', (r*np.cos(theta), r*np.sin(theta), 0)))

    mol.basis = { 'Be': 'cc-pvdz', }
    mol.build(verbose=0)

    mf = scf.RHF(mol)
    mf.verbose = 3
    mf.max_cycle = 1000
    mf.scf(dm0=DMguess)

    DMguess = np.dot( np.dot( mf.mo_coeff, np.diag( mf.mo_occ ) ), mf.mo_coeff.T )

    if ( False ):   
        ccsolver = ccsd.CCSD( mf )
        ccsolver.verbose = 5
        ECORR, t1, t2 = ccsolver.ccsd()
        ECCSD = mf.hf_energy + ECORR
        print "ECCSD for bondlength ",bl," =", ECCSD

    #elif ( bl < 3.35 ):
    else:
        #localization_type = 'meta_lowdin'
        #localization_type = 'boys'
        localization_type = 'iao'
        rotation = np.eye( mol.nao_nr(), dtype=float )
        for i in range(nat):
            theta  = i * (2*np.pi/nat)
            offset = 14 * i # 14 basisfunctions in cc-pVDZ
            # Order of AO: 3s 2p 1d
            rotation[ offset+3:offset+6,  offset+3:offset+6  ] = ringhelper.p_functions( theta )
            rotation[ offset+6:offset+9,  offset+6:offset+9  ] = ringhelper.p_functions( theta )
            rotation[ offset+9:offset+14, offset+9:offset+14 ] = ringhelper.d_functions( theta )
        assert( np.linalg.norm( np.dot( rotation, rotation.T ) - np.eye( rotation.shape[0] ) ) < 1e-6 )
        myInts = localintegrals.localintegrals( mf, range( mol.nao_nr() ), localization_type, rotation )
        if (( localization_type == 'meta_lowdin' ) or ( localization_type == 'iao' )):
            myInts.TI_OK = True
        myInts.molden( 'Be-loc.molden' )

        atoms_per_imp = 1 # Impurity size = 1/2/4 Be atoms
        assert ( nat % atoms_per_imp == 0 )
        orbs_per_imp = myInts.Norbs * atoms_per_imp / nat

        impurityClusters = []
        for cluster in range( nat / atoms_per_imp ):
            impurities = np.zeros( [ myInts.Norbs ], dtype=int )
            for orb in range( orbs_per_imp ):
                impurities[ orbs_per_imp*cluster + orb ] = 1
            impurityClusters.append( impurities )
        if (( localization_type == 'meta_lowdin' ) or ( localization_type == 'iao' )):
            isTranslationInvariant = True
        else:
            isTranslationInvariant = False # Boys TI is not OK
        method = 'CC'
        SCmethod = 'NONE' # NONE or LSTSQ for no self-consistency or least-squares fitting of the u-matrix, respectively
        theDMET = dmet.dmet( myInts, impurityClusters, isTranslationInvariant, method, SCmethod )
        theDMET.doselfconsistent()
        #theDMET.dump_bath_orbs( 'Be-bathorbs.molden' )

