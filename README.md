1. scipy.linalg.eigh

This function is a specialized solver for eigenvalues and eigenvectors.

What it does: It solves the standard eigenvalue problem (Ax=λx) or the generalized eigenvalue problem (Ax=λBx) for complex Hermitian or real symmetric matrices.

Why use it instead of eig?

Speed & Stability: Because it knows the matrix is symmetric (or Hermitian), it uses specialized algorithms (like finding tridiagonal forms) that are faster and numerically more stable than the generic scipy.linalg.eig.

Real Eigenvalues: Symmetric/Hermitian matrices always have real eigenvalues. eigh guarantees the returned eigenvalues are real (no complex components due to floating-point noise).

Sorting: It typically returns the eigenvalues in ascending order.

Note: This function expects dense arrays (standard NumPy arrays). If you try to pass a sparse matrix directly to this function, it may try to convert it to a dense array, potentially running out of memory.

2. scipy.sparse.coo_matrix

This class creates a sparse matrix in COOrdinate format (often called "triplet" or "ijv" format).

What it does: It stores a matrix by keeping lists of the non-zero elements only: (row_index, column_index, value).

Key Characteristics:

Fast Construction: It is the best format for constructing a sparse matrix from scratch. You can just append new coordinates and values to the lists without worrying about the structure.

Duplicate Handling: If you specify multiple entries for the same row/column index, coo_matrix sums them up when converting to other formats. This is incredibly useful for finite element assembly or counting operations.

limitations: It does not support arithmetic operations (like matrix multiplication) or slicing very well.

Typical Workflow: You build your matrix using coo_matrix, then convert it to CSR (Compressed Sparse Row) or CSC (Compressed Sparse Column) format for efficient calculations.


BarElement class
all bars are given a really high stiffness to basically make it so they are rigid. 