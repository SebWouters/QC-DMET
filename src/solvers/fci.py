import numpy as np
import os
import sys
import qcdmet_paths
from pyscf import gto, scf, ao2mo, fci

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, chempot_imp=0.0, printoutput=False ):

    if ( printoutput==False ):
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open('/dev/null', os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)

    FOCKcopy = FOCK.copy()
    if (chempot_imp != 0.0):
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp

    mol = gto.Mole()
    mol.build( verbose=0 )
    mol.atom.append(('H', (0, 0, 0)))
    mol.nelectron = Nel
    mol.incore_anyway = True
    mf = scf.RHF( mol )
    mf.get_hcore = lambda *args: FOCKcopy
    mf.get_ovlp = lambda *args: np.eye( Norb )
    mf._eri = ao2mo.restore(8, TEI, Norb)
    mf.scf()

    assert( Nel % 2 == 0 )
    cisolver = fci.direct_spin0.FCI()
    cisolver.verbose = 0
    cisolver.max_cycle = 200
    cisolver.conv_tol = 1e-12
    EnergyFCI, FCIvector = cisolver.kernel( FOCKcopy, TEI, Norb, Nel, ecore=CONST )
    TwoRDM = cisolver.make_rdm2( FCIvector, Norb, Nel )

    OneRDM = np.einsum( 'ijkk->ij', TwoRDM ) / ( Nel - 1 )

    if ( printoutput==False ):
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)

    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum( 'ij,ij->', OneRDM[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:] )
    ImpurityEnergy += 0.5 * np.einsum( 'ijkl,ijkl->', TwoRDM[:Nimp,:,:,:], TEI[:Nimp,:,:,:] )
    return ( ImpurityEnergy, OneRDM )
