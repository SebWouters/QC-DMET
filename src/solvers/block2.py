import numpy as np
import os
import sys

def solve( CONST, OEI, FOCK, TEI, Norb, Nel, Nimp, chempot_imp=0.0, printoutput=False ):
    """
    Solves the impurity problem using Block2 DMRG.
    Returns (ImpurityEnergy, OneRDM).
    """
    
    # Import block2 here so that QC-DMET can run without it if not using DMRG
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError:
        raise ImportError("Block2 is not installed or cannot be found. Please ensure it is installed in your python environment (e.g. via pip install block2).")

    # Mute standard output if needed
    if not printoutput:
        sys.stdout.flush()
        old_stdout = sys.stdout.fileno()
        new_stdout = os.dup(old_stdout)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, old_stdout)
        os.close(devnull)
    
    FOCKcopy = FOCK.copy()
    if chempot_imp != 0.0:
        for orb in range(Nimp):
            FOCKcopy[ orb, orb ] -= chempot_imp
            
    # QC-DMET normally assumes closed-shell singlets for these impurity solvers
    # Setting SU2 symmetry to explicitly allow Block2 to natively spin-trace the RDMs to match PySCF exactly
    driver = DMRGDriver(scratch=os.path.join(os.getcwd(), '.block2_scratch'), 
                        symm_type=SymmetryTypes.SU2, 
                        n_threads=1)
                        
    driver.initialize_system(n_sites=Norb, n_elec=Nel, spin=0)
    
    # Setup the QC MPO natively (Block2 expects chemist's notation TEI natively)
    mpo = driver.get_qc_mpo(h1e=FOCKcopy, g2e=TEI, iprint=0 if not printoutput else 1)
    
    # Run the DMRG sweeps
    # Defaulting to 250 bond dimension, which is enough for most DMET impurity calculations
    bond_dims = [250, 250, 250, 250, 250]
    noises = [0, 0, 0, 0, 0]
    thrds = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8]
    n_sweeps = len(bond_dims)
    
    ket = driver.get_random_mps(tag='gs', bond_dim=bond_dims[0], nroots=1)
    EnergyDMRG = driver.dmrg(mpo, ket, n_sweeps=n_sweeps, bond_dims=bond_dims, 
                             noises=noises, thrds=thrds, iprint=0 if not printoutput else 1)
                             
    # Extract the full conventional (spin-traced PySCF-format) 1-RDM and 2-RDM
    # In Block2 SU2 mode, get_1pdm() returns the spin-traced 1-RDM identical to PySCF
    rdm1 = driver.get_1pdm(ket)
    # get_2pdm() returns dm[i, j, l, k] = <i^+ j^+ k l>. 
    # PySCF make_rdm2 returns rdm2[i, j, k, l] = <i^+ k^+ l j>.
    # Transposing Block2's get_2pdm via (0, 3, 1, 2) makes it identical to PySCF format
    rdm2 = driver.get_2pdm(ket).transpose(0, 3, 1, 2)

    # Compute evaluating impurity energy exactly like PySCF FCI and CheMPS2
    # Only trace over index 1..Nimp for the first orbital index to localize the energy to the impurity
    ImpurityEnergy = CONST
    ImpurityEnergy += 0.5 * np.einsum('ij,ij->', rdm1[:Nimp,:], OEI[:Nimp,:] + FOCK[:Nimp,:])
    ImpurityEnergy += 0.5 * np.einsum('ijkl,ijkl->', rdm2[:Nimp,:,:,:], TEI[:Nimp,:,:,:])

    # Restore stdout
    if not printoutput:
        sys.stdout.flush()
        os.dup2(new_stdout, old_stdout)
        os.close(new_stdout)
        
    # Return exactly like PySCF FCI and CheMPS2
    return (ImpurityEnergy, rdm1)
