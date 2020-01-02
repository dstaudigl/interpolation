import numpy as np
import sys

class Trigonometric_Polynomial:

    '''Trigonometric Polynomial

    Authors: David Staudigl, Medhi Garouachi
    Date: 02.01.2020

    This class contains the implementation of the interpolation of one -
    dimensional functions by trigonometric polynomials.

    Attributes
    ----------
    a : list
        Contains the polynomial coefficients a.

    b : list
        Contains the polynomial coefficients b.

    N : int
        The degree of the interpolating polynomial.

    Methods
    -------
    __call__( t )
        Evaluates the interpolating polynomial at a given float value or
        numpy.ndarray of float values.
    '''

    def __init__( self, t, f ):

        '''
        Parameters
        ----------
        t : numpy.ndarray | float
            A numpy.ndarray or float value containing the interpolation points.
        f : numpy.ndarray | float
            A numpy.ndarray or float value containing the function evaluations.
        '''
        
        # Set n to be the number of interpolation points:

        n = len( t )

        # Check arguments for validity:

        if n % 2 == 0:
            sys.exit(
                '[{}]: Class {} __init__: Even number of interpolation points.'
                .format( sys.argv[ 0 ], self.__class__.__name__ )
                )

        if len( f ) != n:
            sys.exit(
                '[{}]: Class {} __init__: Size of interpolation points and'\
                'function values does not match.'
                .format( sys.argv[ 0 ], self.__class__.__name__ )
                )

        # Require ascending interpolation points in [ 0, 2 * numpy.pi ):

        if np.all( t[:-1] > t[1:] ) or t[ -1 ] >= 2 * np.pi:
            sys.exit(
                '[{}]: Class {} __init__: Interpolation points not ascending'\
                'in [ 0, 2 * pi).'
                .format( sys.argv[ 0 ], self.__class__.__name__ )
                )

        # Accept either float or numpy.ndarray input:

        if isinstance( t, float ):
            t = [ t ]

        if isinstance( f, float ):
            f = [ f ]

        if not isinstance( t, np.ndarray ):
            sys.exit(
                '[{}]: Class {} __init__: Interpolation points not passed as'\
                'numpy.ndarray or float.'
                .format( sys.argv[ 0 ], self.__class__.__name__ )
                )

        if not isinstance( f, np.ndarray ):
            sys.exit(
                '[{}]: Class {} __init__: Function values not passed as'\
                'numpy.ndarray or float.'
                .format( sys.argv[ 0 ], self.__class__.__name__ )
                )

        # Employ broadcasting to generate the LSE matrix according to task:

        interpolation_matrix_cos = np.cos(
            np.arange( 1, int( ( n + 1 ) / 2 ) ) * t[ :, None ] )
        interpolation_matrix_sin = np.sin(
            np.arange( 1, int( ( n + 1 ) / 2 ) ) * t[ :, None ] )
        interpolation_matrix_const = np.matrix( np.full( ( n, 1 ), 1 / 2 ) )

        interpolation_matrix = np.hstack( (
            interpolation_matrix_const,
            interpolation_matrix_cos,
            interpolation_matrix_sin
            ) )

        # Solve LSE for coefficient list:

        interpolation_coefficients = list(
            np.linalg.solve( interpolation_matrix, f ) )

        # Set the coefficient lists a, b and degree N ( n = 2 * N + 1 ):

        self.a = interpolation_coefficients[ 0 : int( ( n + 1 ) / 2 ) ]
        self.b = interpolation_coefficients[ int( ( n + 1 ) / 2 ) : ]

        self.N = len( self.b )

    def __call__( self, t ):

        '''Returns the evaluation of the interpolating polynomial.

        Parameters
        ----------
        t : numpy.ndarray | float
            A numpy.ndarray or float value containing the evaluation points.

        Returns
        -------
        A numpy.ndarray containing the evaluation of the interpolating
        polynomial at the given points.
        '''

        # Accept either float or numpy.ndarray input:

        if isinstance( t, float ):
            t = [ t ]

        if not isinstance( t, np.ndarray ):
            sys.exit(
                '[{}]: Class {} __call__: Evaluation points not given as'\
                'numpy.ndarray or float.'
                .format( sys.argv[ 0 ], self.__class__.__name__ )
                )

        # Return the polynomial's evaluation at a given value of t:

        return self.a[ 0 ]/2 + sum( [ (
            self.a[ i ] * np.cos( i * t ) + self.b[ i - 1 ] * np.sin( i * t )
            ) for i in range( 1, self.N + 1 ) ] )
