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

def p_functions( theta ):

    # Order of block is 0 : px
    #                   1 : py
    #                   2 : pz
    # [ https://github.com/sunqm/libcint/blob/master/src/cart2sph.c#L25 ]
    rotation = np.eye( 3, dtype=float )
    mycos = np.cos( theta )
    mysin = np.sin( theta )
    rotation[ 0, 0 ] =  mycos
    rotation[ 1, 1 ] =  mycos
    rotation[ 0, 1 ] =  mysin
    rotation[ 1, 0 ] = -mysin
    return rotation
    
def d_functions( theta ):

    # Order of block is 0 : dxy
    #                   1 : dzy
    #                   2 : d(3z^2-x^2-y^2)
    #                   3 : dzx
    #                   4 : d(x^2-y^2)
    # [ https://github.com/sunqm/libcint/blob/master/src/cart2sph.c#L37 ]
    rotation = np.eye( 5, dtype=float )
    mycos  = np.cos( theta )
    mysin  = np.sin( theta )
    mycos2 = mycos * mycos - mysin * mysin # np.cos(2*theta)
    mysin2 = 2 * mycos * mysin             # np.sin(2*theta)
    rotation[ 3, 3 ] =  mycos
    rotation[ 1, 1 ] =  mycos
    rotation[ 3, 1 ] =  mysin
    rotation[ 1, 3 ] = -mysin
    rotation[ 4, 4 ] =  mycos2
    rotation[ 0, 0 ] =  mycos2
    rotation[ 4, 0 ] =  mysin2
    rotation[ 0, 4 ] = -mysin2
    return rotation
    
def f_functions( theta ):

    # Order of block is 0 : f(y(3x^2 - y^2))
    #                   1 : f(xyz)
    #                   2 : f(y(4z^2-x^2-y^2))
    #                   3 : f(z(2z^2-3x^2-3y^2))
    #                   4 : f(x(4z^2-x^2-y^2))
    #                   5 : f(z(x^2-y^2))
    #                   6 : f(x(x^2-3y^2))
    # https://github.com/sunqm/libcint/blob/master/src/cart2sph.c#L72
    rotation = np.eye( 7, dtype=float )
    mycos  = np.cos( theta )
    mysin  = np.sin( theta )
    mycos2 = mycos * mycos - mysin * mysin   # np.cos(2*theta)
    mysin2 = 2 * mycos * mysin               # np.sin(2*theta)
    mycos3 = mycos2 * mycos - mysin2 * mysin # np.cos(3*theta)
    mysin3 = mycos2 * mysin + mysin2 * mycos # np.sin(3*theta)
    rotation[ 4, 4 ] =  mycos
    rotation[ 2, 2 ] =  mycos
    rotation[ 4, 2 ] =  mysin
    rotation[ 2, 4 ] = -mysin
    rotation[ 5, 5 ] =  mycos2
    rotation[ 1, 1 ] =  mycos2
    rotation[ 5, 1 ] =  mysin2
    rotation[ 1, 5 ] = -mysin2
    rotation[ 6, 6 ] =  mycos3
    rotation[ 0, 0 ] =  mycos3
    rotation[ 6, 0 ] =  mysin3
    rotation[ 0, 6 ] = -mysin3
    return rotation
    
    
    
