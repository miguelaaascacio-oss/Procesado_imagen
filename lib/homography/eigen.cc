// ----------------------------------------------------------------------------
// Numerical diagonalization of 3x3 matrcies
// Copyright (C) 2006  Joachim Kopp
// ----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
// ----------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <float.h>
// #include "dsyevc3.h"
// #include "dsyevv3.h"
#include "homography/eigen.h"


double cubic_root( double v ) 
{
	double signo = 1.0;
	double del;

	if( v < 0.0 ) {
		signo = -1.0;
		v *= -1.0;
	}
	del = signo*exp( log(v)/3.0 );

	return( del );
}

/** 
 Args: ax^3 + bx^2 + cx + d = 0
 double * roots must be pointer to an array of size three 
 Returns the number of real root and their values insise roots[]
**/

int cubic( double a, double b, double c, double d, double *roots )
{
	double xN, yN, yN2;
	double delta2, h2, delta;

	xN = -b/(3.0*a);

	yN = xN*( xN*(a*xN + b) + c ) + d;
	yN2 = yN*yN;

	delta2 = (b*b - 3.0*a*c)/(9.0*a*a); 
	h2 = 4.0*a*a*delta2*delta2*delta2;

	if ( fabs(yN2 - h2) < 1e-8 ) {
		if ( fabs( yN2 ) < 1e-8 ){
			// Three equal roots in xN 
			roots[0] = roots[1] = roots[2] = xN;
			return( 3 );
		}
		else {
			// print "Dos reales iguales y la tercera distinta\n";
			delta = cubic_root( yN/(2.0*a) );
			roots[0] = xN + delta;
			roots[1] = roots[2] = xN - 2.0*delta;
			return( 3 );
		}
	}
	else if ( yN2 > h2 ) {
		double v, v1, v2;

		// printf( "Una raiz real\n" );
		v = sqrt( yN2 - h2 );
		v1 = cubic_root( (-yN + v)/(2.0*a) );
		v2 = cubic_root( (-yN - v)/(2.0*a) );

		roots[0] = xN + v1 + v2;
		// print "Raiz en $r1\n";
		return( 1 );
	}
	else {
		double c, s, theta;

		// print "Tres raices reales distintas\n";

		c = -yN/sqrt( h2 );
		s = sqrt( 1.0 - c*c );
		theta = atan2( s, c )/3.0;
		// print "Theta = $theta\n";
		delta = 2.0*sqrt( delta2 );
	
		roots[0] = xN + delta*cos( theta );
		roots[1] = xN + delta*cos( theta + 2.09439510266 );
		roots[2] = xN + delta*cos( theta + 4.18879020533 );
		// print "Raices: $r1 $r2 $r3\n";

		return( 3 );
	}
}

/**
  Calcula los eugenvalores de una matriz simétrica positiva definida A
**/

int eigenValues( double A[][3], double *roots )
{
/**
                                     2                    2              2
(%o5) - s00 s11 s22 + l s11 s22 + s01  s22 + l s00 s22 - l  s22 + s00 s12
        2                      2                    2            2        2
 - l s12  - 2 s01 s02 s12 + s02  s11 + l s00 s11 - l  s11 - l s02  - l s01
    2        3
 - l  s00 + l  = 0
**/

	double b, c, d;

	b = -A[0][0] - A[1][1] - A[2][2];
	c = A[1][1]*A[2][2] + A[0][0]*A[2][2] - A[1][2]*A[1][2] + A[0][0]*A[1][1] - A[0][2]*A[0][2] - A[0][1]*A[0][1]; 
	d = A[0][1]*A[0][1]*A[2][2] + A[0][0]*A[1][2]*A[1][2] - 2.0*A[0][1]*A[0][2]*A[1][2] + A[0][2]*A[0][2]*A[1][1] - A[0][0]*A[1][1]*A[2][2];

	if( cubic( 1.0, b, c, d, roots ) == 3 ) {
		return 3;
	}
	else{
		printf("Esto no debería pasar\n");
		return 0;
	}
}

// ----------------------------------------------------------------------------
int dsyevv3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
  double norm;          // Squared norm or inverse norm of current eigenvector
  double n0, n1;        // Norm of first and second columns of A
  double n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
  double thresh;        // Small number used as threshold for floating point comparisons
  double error;         // Estimated maximum roundoff error in some steps
  double wmax;          // The eigenvalue of maximum modulus
  double f, t;          // Intermediate storage
  int i, j;             // Loop counters
#endif

  // Calculate eigenvalues
  eigenValues(A, w);
  /**norm = w[1];
  w[1] = w[2];
  w[2] = norm;
  */

#ifndef EVALS_ONLY
  wmax = fabs(w[0]);
  if ((t=fabs(w[1])) > wmax)
    wmax = t;
  if ((t=fabs(w[2])) > wmax)
    wmax = t;
  thresh = SQR(8.0 * DBL_EPSILON * wmax);

  // Prepare calculation of eigenvectors
  n0tmp   = SQR(A[0][1]) + SQR(A[0][2]);
  n1tmp   = SQR(A[0][1]) + SQR(A[1][2]);
  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  Q[1][1] = A[0][2]*A[1][0] - A[1][2]*A[0][0];
  Q[2][1] = SQR(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
  A[0][0] -= w[0];
  A[1][1] -= w[0];
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
  norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
  n0      = n0tmp + SQR(A[0][0]);
  n1      = n1tmp + SQR(A[1][1]);
  error   = n0 * n1;
  
  if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
  {
    Q[0][0] = 1.0;
    Q[1][0] = 0.0;
    Q[2][0] = 0.0;
  }
  else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
  {
    Q[0][0] = 0.0;
    Q[1][0] = 1.0;
    Q[2][0] = 0.0;
  }
  else if (norm < SQR(1e-9) * error) // (norm < SQR(64.0 * DBL_EPSILON) * error)
  {                         // If angle between A[0] and A[1] is too small, don't use
    t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
    f = -A[0][0] / A[0][1];
    if (SQR(A[1][1]) > t)
    {
      t = SQR(A[1][1]);
      f = -A[0][1] / A[1][1];
    }
    if (SQR(A[1][2]) > t)
      f = -A[0][2] / A[1][2];
    norm    = 1.0/sqrt(1 + SQR(f));
    Q[0][0] = norm;
    Q[1][0] = f * norm;
    Q[2][0] = 0.0;
  }
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }

  
  // Prepare calculation of second eigenvector
  t = w[0] - w[1];
  if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
  {
    // For non-degenerate eigenvalue, calculate second eigenvector by the formula
    //   v[1] = (A - w[1]).e1 x (A - w[1]).e2
    A[0][0] += t;
    A[1][1] += t;
    Q[0][1]  = Q[0][1] + A[0][2]*w[1];
    Q[1][1]  = Q[1][1] + A[1][2]*w[1];
    Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
    norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
    n0       = n0tmp + SQR(A[0][0]);
    n1       = n1tmp + SQR(A[1][1]);
    error    = n0 * n1;
 
    if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
    {
      Q[0][1] = 1.0;
      Q[1][1] = 0.0;
      Q[2][1] = 0.0;
    }
    else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
    {
      Q[0][1] = 0.0;
      Q[1][1] = 1.0;
      Q[2][1] = 0.0;
    }
    else if (norm < SQR(1e-9) * error) // (norm < SQR(64.0 * DBL_EPSILON) * error)
    {                       // If angle between A[0] and A[1] is too small, don't use
      t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
      f = -A[0][0] / A[0][1];
      if (SQR(A[1][1]) > t)
      {
        t = SQR(A[1][1]);
        f = -A[0][1] / A[1][1];
      }
      if (SQR(A[1][2]) > t)
        f = -A[0][2] / A[1][2];
      norm    = 1.0/sqrt(1 + SQR(f));
      Q[0][1] = norm;
      Q[1][1] = f * norm;
      Q[2][1] = 0.0;
    }
    else
    {
      norm = sqrt(1.0 / norm);
      for (j=0; j < 3; j++)
        Q[j][1] = Q[j][1] * norm;
    }
  }
  else
  {
    // For degenerate eigenvalue, calculate second eigenvector according to
    //   v[1] = v[0] x (A - w[1]).e[i]
    //   
    // This would really get to complicated if we could not assume all of A to
    // contain meaningful values.
    A[1][0]  = A[0][1];
    A[2][0]  = A[0][2];
    A[2][1]  = A[1][2];
    A[0][0] += w[0];
    A[1][1] += w[0];
    for (i=0; i < 3; i++)
    {
      A[i][i] -= w[1];
      n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
      if (n0 > thresh)
      {
        Q[0][1]  = Q[1][0]*A[2][i] - Q[2][0]*A[1][i];
        Q[1][1]  = Q[2][0]*A[0][i] - Q[0][0]*A[2][i];
        Q[2][1]  = Q[0][0]*A[1][i] - Q[1][0]*A[0][i];
        norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
        if (norm > SQR(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
        {                                         // the two vectors was not too small
          norm = sqrt(1.0 / norm);
          for (j=0; j < 3; j++)
            Q[j][1] = Q[j][1] * norm;
          break;
        }
      }
    }
    
    if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
    {
      for (j=0; j < 3; j++)
        if (Q[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
        {                                                     // ... and swap it with the next one
          norm          = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j+1)%3][0]));
          Q[j][1]       = Q[(j+1)%3][0] * norm;
          Q[(j+1)%3][1] = -Q[j][0] * norm;
          Q[(j+2)%3][1] = 0.0;
          break;
        }
    }
  }
      
  
  // Calculate third eigenvector according to
  //   v[2] = v[0] x v[1]
  Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
  Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
  Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];
#endif

  return 0;
}

