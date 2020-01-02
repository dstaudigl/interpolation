import numpy as np
import sys

class Cubic_Periodic_Spline:

    '''Cubic Periodic Spline

    Authors: David Staudigl, Medhi Garouachi
    Date: 02.01.2020

    This class contains the implementation of the interpolation of one -
    dimensional functions by cubic periodic splines of periodicity 2 * pi.

    Attributes
    ----------
    n : int
        The number of interpolation points.
    t : list
        A list containing the interpolation points, including the concluding
        point x_n = x_0 + 2 * pi.
    f : list
        A list containing the function evaluation at the interpolation points,
        including the concluding evaluation f_n = f_0.
    h : list
        A list containing the distance between neighbouring interpolation
        points.
    m : list
        A list containing the moments as calculated by the interpolating
        algorithm.
    C : list
        A list containing the integration constants C as calculated by the
        interpolating algorithm.
    D : list
        A list containing the integration constants D as calculated by the
        interpolating algorithm.

    Methods
    -------
    __call__( t )
        Evaluates the interpolating spline at a given float value or
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

        # Set n to be the number of interpolation points and check for validity:

        self.n = len( t )

        if self.n % 2 == 0:
            sys.exit(
                '[{}]: Class {} __init__: Even number of interpolation points.'
                .format( sys.argv[ 0 ], self.__class__.__name__ )
                )

        if len( f ) != self.n:
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

        # Set t and f lists to retain for evaluation:

        self.t = list( t )
        self.f = list( f )

        # Append concluding interval with periodicity 2 * numpy.pi;
        # lists to be of length n + 1:

        self.t.append( 2 * np.pi + self.t[ 0 ] )
        self.f.append( f[ 0 ] )

        # Generate h, my, l and d lists:

        # h_0 .. h_(n-1) to be retained for evaluation:

        self.h = list( np.diff( self.t ) )

        my = []

        # my_1 .. my_(n-1):

        for a, b in zip( self.h[ 0 : ], self.h[ 1 : ] ):
            my.append( a / ( a + b ) )

        # my_n:

        my.append( self.h[ -1 ] / ( self.h[ -1 ] + self.h[ 0 ] ) )

        # l_1 .. l_n:

        l = [ 1 - a for a in my ]

        # d_1 .. d_(n-1):

        d = []

        for i in range( 1, self.n ):
            d.append(
                (
                    ( self.f[ i + 1 ] - self.f[ i ] ) / self.h[ i ]
                    - ( self.f[ i ] - self.f[ i - 1 ] ) / self.h[ i - 1 ]
                ) / ( self.h[ i - 1 ] + self.h[ i ] ) * 6
                )

        # d_0:

        d.append(
            (
                ( self.f[ 1 ] - self.f[ 0 ] ) / self.h[ 0 ]
                - ( self.f[ -1 ] - self.f[ -2 ] ) / self.h[ - 1 ]
            ) / ( self.h[ 0 ] + self.h[ -1 ] ) * 6
            )

        # Generate (n-1) x (n-1) LSE matrix by rolling and adding:

        interpolation_matrix_two = np.diag( 2 * np.ones( self.n ) )
        interpolation_matrix_l = np.roll( np.diag( l ), 1, axis = 1 )
        interpolation_matrix_my = np.roll( np.diag( my ), 1, axis = 0 )

        interpolation_matrix = ( interpolation_matrix_two
            + interpolation_matrix_l
            + interpolation_matrix_my )

        # Solve LSE for moment list and roll:

        self.m = list( np.roll(
            np.linalg.solve( interpolation_matrix, d ), 1 ) )

        # Append concluding moment:

        self.m.append( self.m[ 0 ] )

        # Compute the coefficient lists C and D:

        self.C = []
        self.D = []

        for i in range( 0, self.n ):
            self.C.append(
                ( self.f[ i + 1 ] + self.f[ i ] ) / 2
                - self.h[ i ] ** 2 / 12 * ( self.m[ i + 1 ] + self.m[ i ] )
                )
            self.D.append(
                ( self.f[ i + 1 ] - self.f[ i ] ) / self.h[ i ]
                - self.h[ i ] / 6 * ( self.m[ i + 1 ] - self.m[ i ] )
                )

    def __call__( self, t ):

        '''Returns the evaluation of the interpolating spline.

        Parameters
        ----------
        t : numpy.ndarray | float
            A numpy.ndarray or float value containing the evaluation points.

        Returns
        -------
        A numpy.ndarray containing the evaluation of the interpolating
        spline at the given points.
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

        # Use numpy.where to find the corresponding interval index for each t:

        t_matrix = np.tile( t, ( self.n, 1 ) )
        T_matrix = np.tile( self.t[ : -1 ], ( len( t ), 1 ) ).T

        indices = ( np.sum(
            np.where( t_matrix >= T_matrix, 1, 0 ), axis = 0
            ) - 1 ) % self.n

        # Evaluate for each t:

        return [
            self.C[ indices[ i ] ]
            + self.D[ indices[ i ] ]
                * ( t[ i ] - (
                    self.t[ indices[ i ] ] + self.t[ indices[ i ] + 1 ]
                  ) / 2 )
            + (
                self.m[ indices[ i ] + 1 ]
                    * ( t[ i ] - self.t[ indices[ i ] ] ) ** 3
                - self.m[ indices[ i ] ]
                    * ( t[ i ] - self.t[ indices[ i ] + 1 ] ) ** 3
            ) / 6 / self.h[ indices[ i ] ]
            for i in range( 0, len( t ) ) ]
