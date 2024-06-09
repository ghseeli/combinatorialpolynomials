# ****************************************************************************
#  Copyright (C) 2024 George H. Seelinger <ghseeli@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
# ***************************************************************************


from sage.all import IntegerVectors, vector, matrix, QQ

def monomial_basis(n, br):
    r"""
    Return monomial basis of polynomials in ``br`` of nonnegative degree ``n``.
    """
    gens = br.gens()
    vecs = IntegerVectors(n, length=len(gens))
    return [prod([gens[i]**(vec[i]) for i in range(len(gens))]) for vec in vecs]

def encode_fn_to_vec_with_monomial_encoding(fn, encoding, base_ring=QQ):
    r"""
    Given a polynomial ``fn`` and a dictionary ``encoding`` mapping monomials to coordinates, return a vector representing the polynomial.

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: poly = x1^3+x1*x2^2
        sage: encoding = {x1^3:0, x1^2*x2:1, x1*x2^2:2, x2^3:3}
        sage: encode_fn_to_vec_with_monomial_encoding(poly, encoding, R)
        (1, 0, 1, 0)
    """
    list_vec = [0]*len(encoding)
    for (coeff, mon) in list(fn):
        list_vec[encoding[mon]] = coeff
    return vector(base_ring, list_vec)

def polys_to_matrix(fns, base_ring=QQ, mons=None):
    r"""
    Given a list of polynomials ``fns``, return a matrix represnting the polynomials, each polynomial as a row.

    Note, the matrix is only supported on the monomials present in the given functions, so it will not have any zero columns.

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: polys = [x1,x1+x2,x1+x2+x3]
        sage: polys_to_matrix(polys)
        [1 0 0]
        [1 1 0]
        [1 1 1]
    """
    par = parent(fns[0])
    if not mons:
        mons = list(reversed(sorted(list(set([mon for fn in fns for mon in fn.monomials()])))))
    encoding = {mons[i]:i for i in range(len(mons))}
    mat = matrix(base_ring, [encode_fn_to_vec_with_monomial_encoding(fn, encoding, base_ring) for fn in fns])
    return mat

def solve_polynomial_in_terms_of_basis(fn, basis, base_ring=QQ):
    r"""
    Given a polynomial ``fn`` with coefficients in ``base_ring``, return the coefficients of its expansion into ``basis`` as a list.

    This method works via linear algebra row-reduction and is not optimized. 
    """
    leading_basis = basis[0]
    par = leading_basis.parent()
    fns = basis + [fn]
    mat = polys_to_matrix(fns, base_ring)
    reduced_mat = mat.transpose().rref()
    for row in reduced_mat:
        if row[-1] != 0 and list(row)[:-1] == [0]*(len(row)-1):
            raise Exception("Given function was not a linear combination of homogeneous basis!")
    for i in range(len(basis)):
        if reduced_mat[i][i] != 1:
            raise Exception("Matrix did not row reduce as expected!")
    return [reduced_mat[i][-1] for i in range(len(basis))]
