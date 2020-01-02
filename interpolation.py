#!/usr/bin/env python3

'''2 - Dimensional Interpolation

Authors: David Staudigl, Medhi Garouachi
Date: 02.01.2020

This script applies various methods of component - wise interpolation for
2 - dimensional parametric functions:

    * Equidistant trigonometric polynomials
    * Equidistant cubic periodic splines
    * Non - equidistant cubic periodic splines

The user must define ideally 2 * pi - periodic functions to be interpolated and
may set the number of interpolation and evaluation points as well as choose
whether to display or save the output.

The relevant setup section is marked in the source code.

The classes Trigonometric_Polynomial and Cubic_Periodic_Spline containing the
implementation of the one - dimensional interpolation are imported from the
classes directory.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

from classes.trigonometric_polynomials import *
from classes.cubic_periodic_splines import *

def main():

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                 Begin Setup                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Define the X component of the function to be interpolated:

    def f_x( t ):

        # Example function:
        return 0.3 * np.cos( t ) + 0.15 * np.cos( 2 * t )

    # Define the Y component of the function to be interpolated:

    def f_y( t ):

        # Example function:
        return 0.3 * np.sin( t )

    # Set N to be the odd number of interpolation points:

    N = 9

    # Set E to be the number of plot evaluation points
    # ( less than 10^6 suggested, depending on the system ):

    E = 100000

    # Set SVG to True to save plots in SVG format:

    SVG = True

    # Set SHOW to True to show plots:

    SHOW = True

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                   End Setup                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Falsify the validity of user settings:

    if N % 2 == 0 or N <= 0:
        sys.exit( '[{}]: Invalid number of interpolation points.'
            .format( sys.argv[ 0 ] ) )

    if E <= 0:
        sys.exit( '[{}]: Invalid number of plot evaluation points.'
            .format( sys.argv[ 0 ] ) )

    if SVG == False and SHOW == False:
        sys.exit( '[{}]: Both output methods declined.'
            .format( sys.argv[ 0 ] ) )

    # Generate interpolation points 0 .. N-1 and function evaluation:

    interpolation_t = np.linspace( 0, 2 * np.pi, N, endpoint = False )

    interpolation_x = f_x( interpolation_t )
    interpolation_y = f_y( interpolation_t )

    # Generate trigonometric polynomials:

    q_x = Trigonometric_Polynomial( interpolation_t, interpolation_x )
    q_y = Trigonometric_Polynomial( interpolation_t, interpolation_y )

    # Generate cubic periodic splines:

    s_x = Cubic_Periodic_Spline( interpolation_t, interpolation_x )
    s_y = Cubic_Periodic_Spline( interpolation_t, interpolation_y )

    # Generate non-equidistant cubic periodic splines:

    neq_interpolation_t = np.concatenate( (
        [ 0 ], np.cumsum( (
            np.diff(
                np.concatenate( ( interpolation_x, [ interpolation_x[ 0 ] ] ) )
            ) ** 2
            + np.diff(
                np.concatenate( ( interpolation_y, [ interpolation_y[ 0 ] ] ) )
            ) ** 2
        ) ** 0.5 )
        ) )

    # Scale to [ 0, 2*numpy.pi ] and remove concluding element:

    neq_interpolation_t = (
        neq_interpolation_t / neq_interpolation_t[ -1 ] * 2 * np.pi
    )[ : -1 ]

    neq_s_x = Cubic_Periodic_Spline( neq_interpolation_t, interpolation_x )
    neq_s_y = Cubic_Periodic_Spline( neq_interpolation_t, interpolation_y )

    # Generate evaluation points and interpolation evaluations:

    t = np.linspace( 0, 2 * np.pi, E, endpoint = True )

    q_x_t = q_x( t )
    q_y_t = q_y( t )

    s_x_t = s_x( t )
    s_y_t = s_y( t )

    neq_s_x_t = neq_s_x( t )
    neq_s_y_t = neq_s_y( t )

    f_x_t = f_x( t )
    f_y_t = f_y( t )

    # Create figures and subplots for component and parametric plots:

    fig_x, ( ( ax_x, ax_neq_x) ) = plt.subplots( 2, 1, figsize = ( 9, 9 ) )
    fig_y, ( ( ax_y, ax_neq_y) ) = plt.subplots( 2, 1, figsize = ( 9, 9 ) )

    fig_p, ( ( ax_p ) ) = plt.subplots( 1, 1, figsize = ( 9, 9 ) )

    # Plot data in subplots:

    ax_x.plot( t, q_x_t, '-', color = 'royalblue', label = '$q_x( t )$' )
    ax_x.plot( t, s_x_t, '-', color = 'orangered', label = '$s_x( t )$' )
    ax_x.plot( t, f_x_t, ':', color = 'mediumturquoise', label = '$f_x( t )$' )
    ax_x.plot( interpolation_t, interpolation_x, 'o', markersize = 6, color = 'blue', label = '$int$' )

    ax_y.plot( t, q_y_t, '-', color = 'royalblue', label = '$q_y( t )$' )
    ax_y.plot( t, s_y_t, '-', color = 'orangered', label = '$s_y( t )$' )
    ax_y.plot( t, f_y_t, ':', color = 'mediumturquoise', label = '$f_y( t )$' )
    ax_y.plot( interpolation_t, interpolation_y, 'o', markersize = 6, color = 'blue', label = '$int$' )

    ax_neq_x.plot( t, neq_s_x_t, '-', color = 'indianred', label = '$n_x( t )$' )
    ax_neq_x.plot( neq_interpolation_t, interpolation_x, 'o', markersize = 6, color = 'blue', label = '$int$' )

    ax_neq_y.plot( t, neq_s_y_t, '-', color = 'indianred', label = '$n_y( t )$' )
    ax_neq_y.plot( neq_interpolation_t, interpolation_y, 'o', markersize = 6, color = 'blue', label = '$int$' )

    ax_p.plot( q_x_t, q_y_t, '-', color = 'royalblue', label = '$q( t )$' )
    ax_p.plot( s_x_t, s_y_t, '-', color = 'orangered', label = '$s( t )$' )
    ax_p.plot( neq_s_x_t, neq_s_y_t, '-', color = 'indianred', label = '$n( t )$' )
    ax_p.plot( f_x_t, f_y_t, ':', color = 'mediumturquoise', label = '$f( t )$' )
    ax_p.plot( interpolation_x, interpolation_y, 'o', markersize = 6, color = 'blue', label = '$int$' )

    # Set xticks to radians for component plots:

    ticks = [ t * np.pi for t in np.linspace( 0, 2, 9, endpoint = True ) ]
    ticklabels = [ format( t, '.3g' ) for t in np.linspace( 0, 2, 9, endpoint = True ) ]

    ax_x.set_xticks( ticks )
    ax_x.set_xticklabels( ticklabels )

    ax_y.set_xticks( ticks )
    ax_y.set_xticklabels( ticklabels )

    ax_neq_x.set_xticks( ticks )
    ax_neq_x.set_xticklabels( ticklabels )

    ax_neq_y.set_xticks( ticks )
    ax_neq_y.set_xticklabels( ticklabels )

    # Set descriptions for all plots:

    ax_x.legend()
    ax_x.set_title( 'Equidistant interpolation of $f_x$:', fontsize = 12 )
    ax_x.set_xlabel( '$t\ [rad/\pi]$' )
    ax_x.set_ylabel( '$y( t )\ [1]$' )

    ax_y.legend()
    ax_y.set_title( 'Equidistant interpolation of $f_y$:', fontsize = 12 )
    ax_y.set_xlabel( '$t\ [rad/\pi]$' )
    ax_y.set_ylabel( '$y( t )\ [1]$' )

    ax_neq_x.legend()
    ax_neq_x.set_title( 'Non-equidistant interpolation of $f_x$:', fontsize = 12 )
    ax_neq_x.set_xlabel( '$t\ [rad/\pi]$' )
    ax_neq_x.set_ylabel( '$y( t )\ [1]$' )

    ax_neq_y.legend()
    ax_neq_y.set_title( 'Non-equidistant interpolation of $f_y$:', fontsize = 12 )
    ax_neq_y.set_xlabel( '$t\ [rad/\pi]$' )
    ax_neq_y.set_ylabel( '$y( t )\ [1]$' )

    ax_p.legend()
    ax_p.set_title( 'Interpolation of $f$:', fontsize = 12 )
    ax_p.set_xlabel( '$x( t )\ [1]$' )
    ax_p.set_ylabel( '$y( t )\ [1]$' )

    # Handle subplot margins:

    fig_x.tight_layout()
    fig_y.tight_layout()

    # Display and / or save both figures:

    if SVG == True:
        fig_x.savefig( 'plots/f_x.svg', format = 'svg' )
        fig_y.savefig( 'plots/f_y.svg', format = 'svg' )
        fig_p.savefig( 'plots/f.svg', format = 'svg' )

    if SHOW == True:
        plt.show()

if __name__ == '__main__':
        main()
