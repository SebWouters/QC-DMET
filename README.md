QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
===================================================================================================

This is a modernized fork of [sebwouters/qc-dmet](https://github.com/sebwouters/qc-dmet), updated for Python 3.10+ and PySCF 2.0+.
--------------------
Copyright (C) 2015 Sebastian Wouters <sebastianwouters@gmail.com>

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


Building and testing
--------------------

**Environment setup (Python 3.10 + PySCF 2.0+):**

    > module load miniconda3/24.1.2-py310
    > conda create -n qcdmet python=3.10 -y
    > conda activate qcdmet
    > conda install -y numpy scipy pyscf -c conda-forge

QC-DMET requires numpy, scipy, and [pyscf](https://github.com/sunqm/pyscf).
The `ED`/`FCI` solver uses PySCF's built-in `pyscf.fci` module.
[Block2](https://github.com/block-hczhai/block2-preview) via `pip` is optional and only required for the `DMRG` solver.
[CheMPS2](https://github.com/SebWouters/CheMPS2) is supported via the `DMRG-CheMPS2` keyword for legacy DMRG.

The path to pyscf can be adjusted in [qcdmet_paths.py](src/qcdmet_paths.py).

Compile the C++ library libqcdmet.so (requires Intel compilers and MKL):

    > module load intel/2021.10.0 intel-oneapi-mkl/2023.2.0
    > cd lib
    > mkdir build
    > cd build
    > CXX=icpc CC=icc cmake .. -DMKL=ON
    > make
    > cd ../..

Start from the files examples/*.py.

Performance testing
-------------------

### 1. Find the most costly functions:

    > python -m cProfile -o testx.profile testx.py
    > python -m pstats testx.profile
    >>> sort cumulative
    >>> stats

### 2. Find what makes them most costly:

Place just before the function you want to profile @profile:

    @profile
    def construct1RDM_loc_response( self, doSCF, umat, list_H1 ):

And then use [line_profiler](https://github.com/rkern/line_profiler):

    > kernprof -lv testx.py

