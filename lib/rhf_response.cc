/*
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
*/

#include <cstdlib>

extern "C" {

    void dgemm_(char * transA, char * transB, const int * m, const int * n, const int * k,
                double * alpha, double * A, const int * lda, double * B, const int * ldb, double * beta, double * C, const int * ldc);
    void dsyev_(char * jobz, char * uplo, const int * n, double * A, const int * lda, double * W, double * work, int * lwork, int * info);
    void dcopy_(const int * n, double * x, int * incx, double * y, int * incy);

}

extern "C"{
void rhf_response(const int Norb, const int Nterms, const int numPairs, int * H1start, int * H1row, int * H1col, double * H0, double * rdm_deriv){

    const int size = Norb * Norb;
    const int nVir = Norb - numPairs;

    double * eigvecs = (double *) malloc(sizeof(double)*size);
    double * eigvals = (double *) malloc(sizeof(double)*Norb);
    double * temp    = (double *) malloc(sizeof(double)*nVir*numPairs);

    // eigvecs and eigvals contain the eigenvectors and eigenvalues of H0
    {
        int inc = 1;
        dcopy_( &size, H0, &inc, eigvecs, &inc );
        char jobz = 'V';
        char uplo = 'U';
        int info;
        int lwork = 3*Norb-1;
        double * work = (double *) malloc(sizeof(double)*lwork);
        dsyev_( &jobz, &uplo, &Norb, eigvecs, &Norb, eigvals, work, &lwork, &info );
        free(work);
    }

    double * occ  = eigvecs;
    double * virt = eigvecs + numPairs * Norb;

    // H0 contains the 1-RDM of the RHF calculation: H0 = 2 * OCC * OCC.T
    {
        char tran = 'T';
        char notr = 'N';
        double alpha = 2.0;
        double beta  = 0.0;
        dgemm_( &notr, &tran, &Norb, &Norb, &numPairs, &alpha, occ, &Norb, occ, &Norb, &beta, H0, &Norb );
    }

    // temp[ vir + nVir * occ ] = - 1 / ( eps_vir - eps_occ )
    for ( int orb_vir = 0; orb_vir < nVir; orb_vir++ ){
        for ( int orb_occ = 0; orb_occ < numPairs; orb_occ++ ){
            temp[ orb_vir + nVir * orb_occ ] = - 1.0 / ( eigvals[ numPairs + orb_vir ] - eigvals[ orb_occ ] );
        }
    }

    #pragma omp parallel
    {
        double * work1 = (double *) malloc(sizeof(double)*size);
        double * work2 = (double *) malloc(sizeof(double)*Norb*numPairs);

        #pragma omp for schedule(static)
        for ( int deriv = 0; deriv < Nterms; deriv++ ){

            // work1 = - VIRT.T * H1 * OCC / ( eps_vir - eps_occ )
            for ( int orb_vir = 0; orb_vir < nVir; orb_vir++ ){
                for ( int orb_occ = 0; orb_occ < numPairs; orb_occ++ ){
                    double value = 0.0;
                    for ( int elem = H1start[ deriv ]; elem < H1start[ deriv + 1 ]; elem++ ){
                        value += virt[ H1row[ elem ] + Norb * orb_vir ] * occ[ H1col[ elem ] + Norb * orb_occ ];
                    }
                    work1[ orb_vir + nVir * orb_occ ] = value * temp[ orb_vir + nVir * orb_occ ];
                }
            }

            // work1 = 2 * VIRT * work1 * OCC.T
            {
                char notr = 'N';
                double alpha = 2.0;
                double beta = 0.0;
                dgemm_( &notr, &notr, &Norb, &numPairs, &nVir, &alpha, virt, &Norb, work1, &nVir, &beta, work2, &Norb ); // work2 = 2 * VIRT * work1
                alpha = 1.0;
                char tran = 'T';
                dgemm_( &notr, &tran, &Norb, &Norb, &numPairs, &alpha, work2, &Norb, occ, &Norb, &beta, work1, &Norb ); // work1 = work2 * OCC.T
            }

            // rdm_deriv[ row + Norb * ( col + Norb * deriv ) ] = work1 + work1.T
            for ( int row = 0; row < Norb; row++ ){
                for ( int col = 0; col < Norb; col++ ){
                    rdm_deriv[ row + Norb * ( col + Norb * deriv ) ] = work1[ row + Norb * col ] + work1[ col + Norb * row ];
                }
            }
        }

        free(work1);
        free(work2);
    }

    free(temp);
    free(eigvals);
    free(eigvecs);

}
}

