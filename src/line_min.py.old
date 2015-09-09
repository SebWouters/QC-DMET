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

import numpy as np
import scipy.optimize

#
# Based on Gerald's DMET for Hubbard version 20120420
# http://www.princeton.edu/chemistry/chan/software/dmet/
#

def optimize( func, grad, param, max_iter=20 ):

    for it in range( max_iter ):
    
        # Get the cost function value in 'costf'
        resid = func( param )
        costf = np.einsum( 'i,i->', resid, resid )
        nr    = len( resid )
        nx    = len( param )
        
        # Find the restricted step size according to the gradient
        # Use thereto the augmented linear equation (Eq. 272 of helpers.py from Gerald)
        g2 = np.zeros( [nr+nx,nx], dtype=float )
        r2 = np.zeros( [nr+nx], dtype=float )
        if ( grad == None ):
            stepsize = 1e-5
            for counter in range(nx):
                dx = np.zeros([nx], dtype=float)
                dx[counter] = stepsize
                g2[:nr,counter] = ( func( param + dx ) - func( param - dx ) ) / ( 2 * stepsize )
        else:
            g2[:nr,:] = grad( param )
        g2[nr:,:] = 0.1 * np.sqrt( costf ) * np.eye( nx )
        r2[:nr]   = resid
        dx, residuals, rank, sigma = np.linalg.lstsq( g2, r2 )
        
        def LineSearchFunc( step ):
            resid1 = func( param - step * dx )
            costf1 = np.einsum( 'i,i->', resid1, resid1 )
            return costf1
        
        def FindStep():
            grid = list( np.arange( -0.0, 2.00001, 0.2 ) )
            fval = [ LineSearchFunc( step ) for step in grid ]
            sval = grid[ np.argmin( fval ) ]
            if ( abs( sval ) > 1e-4 ):
                return sval
            else:
                return scipy.optimize.fmin( func=LineSearchFunc, x0=np.array([0.001]), disp=0, xtol=1e-10 )
        
        mystep = FindStep()
        
        if ( abs( mystep ) * np.linalg.norm( dx ) < 1e-10 ):
            break
        
        param -= mystep * dx
        
    # After num_iter iterations, return the current value of param
    return param
    
