QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
===================================================================================================

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

QC-DMET requires python, numpy, scipy, CMake,
[libchemps2](https://github.com/SebWouters/CheMPS2), 
[libcint](https://github.com/sunqm/libcint),
[pyscf](https://github.com/sunqm/pyscf),
[psi4](https://github.com/psi4/psi4public), and
[psi4cc](https://github.com/SebWouters/psi4cc).

The path to PyCheMPS2.so can be adjusted in [chemps2.py](src/chemps2.py),
the path to _psi4.so can be adjusted in [psi4cc.py](src/psi4cc.py), and
the path to the folder in which pyscf is installed in
[localintegrals.py](src/localintegrals.py).

Go to the folder lib and compile libqcdmet.so:

    > cd lib
    > mkdir build
    > cd build
    > CXX=icpc CC=icc cmake .. -DMKL=ON
    > make
    > cd ../..

Start from the files tests/test*.py.

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

