# SVD Image Compression

This repository contains an implementation of image compression using Singular Value Decomposition (SVD). The algorithm is developed in MATLAB and includes advanced matrix manipulation techniques, such as QR factorization, Givens rotations, Wilkinson shift, and Hessenberg reduction.
## Implementation
The implementation consists of several key steps, all developed in MATLAB:
# 1️⃣ Hessenberg Matrix Reduction

The first step in computing the eigenvalues of a matrix is reducing it to Hessenberg form. This is achieved using Householder transformations, which eliminate the elements below the first subdiagonal.

📌 Code: hessemberg.m

    Takes a matrix AA as input and transforms it into a Hessenberg matrix.

    Uses Householder transformations to remove unnecessary coefficients.

2️⃣ Eigenvalue Computation via QR Factorization

To determine the eigenvalues of the matrix, iterative QR factorization is applied, gradually converging to a diagonal matrix containing the eigenvalues on the main diagonal.

📌 Code: qrfatt.m

    Implements QR decomposition using Givens rotations.

    Accelerates convergence with Wilkinson shift.

    Tracks applied transformations, storing them in matrix QQ.

📌 Code: givensRotations.m

    Computes the Givens rotations required to eliminate elements below the main diagonal.

3️⃣ Singular Value Decomposition (SVD)

Once the eigenvalues are obtained, SVD decomposition is performed, factorizing the matrix into three components:
A=UΣVT
A=UΣVT

📌 Code: my_svd.m

    Computes the SVD using the previously computed results.

    Extracts matrices U,Σ,VU,Σ,V from the eigenvalues and eigenvectors of AATAAT.

4️⃣ Image Compression using SVD

The compression algorithm is applied separately to the three channels (R, G, B) of an RGB image. By reducing the number of retained singular values, efficient compression is achieved with minimal quality loss.

📌 Code: image_compression.m

    Performs SVD on each image channel (R, G, B).

    Reconstructs the image using only the first kk singular values.

    Computes the compression ratio, highlighting memory savings.

    Displays reconstructed images at different kk values.

📄 Documentation

A detailed explanation of the implemented method, along with theoretical foundations, is provided in:

📌 /doc/presentation.pdf

This document includes:
✔️ Mathematical definition of SVD decomposition.
✔️ Explanation of the compression method through singular value truncation.
✔️ Details on the QR algorithm and Givens rotations.
✔️ Analysis of the trade-off between reconstruction quality and storage efficiency.
