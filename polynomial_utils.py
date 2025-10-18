# ****************************************************************************
#  Copyright (C) 2024 George H. Seelinger <ghseeli@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
# ***************************************************************************


from sage.all import IntegerVectors, vector, matrix, QQ, PolynomialRing, LaurentPolynomialRing, parent, prod
from sage.rings.polynomial.multi_polynomial_sequence import PolynomialSequence

def generate_polynomial_ring(br, num_vars, x_pref='x', start=1):
    r"""
    Return a polynomial ring of ``num_vars`` variables adjoined to ``br``.
    """
    xvars = [x_pref+str(i+start) for i in range(num_vars)]
    return PolynomialRing(br, xvars, num_vars)

def generate_laurent_polynomial_ring(br, num_vars, x_pref='x', start=1):
    r"""
    Return a Laurent polynomial ring of ``num_vars`` variables adjoined to ``br``.
    """
    xvars = [x_pref+str(i+start) for i in range(num_vars)]
    return LaurentPolynomialRing(br, xvars)

def generate_multi_polynomial_ring(br, num_vars, prefs=['x','y'], start=1):
    r"""
    Return a polynomial ring of ``num_vars`` variables in each of ``prefs`` adjoined to ``br``.
    """
    xyvars = [pref+str(i+start) for pref in prefs for i in range(num_vars)]
    return PolynomialRing(br, xyvars, num_vars*len(prefs))

def monomial_basis(n, br):
    r"""
    Return monomial basis of polynomials in ``br`` of nonnegative degree ``n``.

    EXAMPLES::

        sage: R = QQ['x1,x2,x3']
        sage: monomial_basis(2,R)
        [x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2]
        sage: R = QQ['x0,x1,x2']
        sage: monomial_basis(1,R)
        [x0, x1, x2]
        sage: monomial_basis(0,R)
        [1]
    """
    gens = br.gens()
    vecs = IntegerVectors(n, length=len(gens))
    return [prod([gens[i]**(vec[i]) for i in range(len(gens))]) for vec in vecs]

def monomial_basis_in_fixed_xy_degree(m, n, br, x_pref='x', y_pref='y'):
    r"""
    Return monomial basis of polynomials in ``br`` of nonnegative degree ``m+n`` where the `x`-degree of the monomials is ``m`` and the `y`-degree is ``n``.

    EXAMPLES::

        sage: R = generate_multi_polynomial_ring(QQ,2)
        sage: len(monomial_basis_in_fixed_xy_degree(2,2,R))
        9
   """
    gens = br.gens()
    xx = [g for g in gens if str(g)[0] == x_pref]
    yy = [g for g in gens if str(g)[0] == y_pref]
    xxyy = xx+yy
    vecs = [vec for vec in IntegerVectors(m+n, length=len(xx)+len(yy)) if sum(vec[:len(xx)]) == m and sum(vec[len(xx):]) == n]
    return [prod([xxyy[i]**(vec[i]) for i in range(len(xxyy))]) for vec in vecs]

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

def polys_to_matrix(fns, base_ring=QQ, mons=None, sparse=False):
    r"""
    Given a list of polynomials ``fns``, return a matrix represnting the polynomials, each polynomial as a row.

    Note, when ``mons`` is ``None``, the matrix is only supported on the monomials present in the given functions, so it will not have any zero columns.

    Also, the ``coefficients_monomials()`` method for the ``Sequence`` object in Sage has a similar functionality, and may be more optimal in general.

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: polys = [x1,x1+x2,x1+x2+x3]
        sage: polys_to_matrix(polys)
        [1 0 0]
        [1 1 0]
        [1 1 1]
        sage: polys_to_matrix(polys, sparse=True)
        [1 0 0]
        [1 1 0]
        [1 1 1]
        sage: polys_to_matrix(polys, sparse=True).is_sparse()
        True
    """
    par = parent(fns[0])
    if not mons:
        mons = list(reversed(sorted(list(set([mon for fn in fns for mon in fn.monomials()])))))
    if sparse:
        mat = PolynomialSequence(par, fns).coefficients_monomials(order=mons)[0]
        return mat
    else:
        encoding = {mons[i]:i for i in range(len(mons))}
        mat = matrix(base_ring, [encode_fn_to_vec_with_monomial_encoding(fn, encoding, base_ring) for fn in fns])
        return mat

def solve_polynomial_in_terms_of_basis(fn, basis, base_ring=QQ, sparse=False):
    r"""
    Given a polynomial ``fn`` with coefficients in ``base_ring``, return the coefficients of its expansion into ``basis`` as a list.

    This method works via linear algebra row-reduction and is not optimized. 

    Also, ``fn`` needs only be in the ``base_ring``-span of ``basis``, but the elements of ``basis`` must be linearly-independent.

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: basis = [x1+x2,x1-x2,x3^2]
        sage: solve_polynomial_in_terms_of_basis(x1, basis)
        [1/2, 1/2, 0] 
        sage: solve_polynomial_in_terms_of_basis(x2+x3^2, basis)
        [1/2, -1/2, 1]
        sage: A = Frac(QQ['q1,q2,q3'])['x1,x2,x3']
        sage: from schubert_polynomials import quantum_Schubert
        sage: solve_polynomial_in_terms_of_basis(A(quantum_Schubert([2,3,1])),[A.one(),A('x1*x2')],base_ring=A.base_ring())
        [q1, 1]
    """
    leading_basis = basis[0]
    par = leading_basis.parent()
    fns = basis + [fn]
    mat = polys_to_matrix(fns, base_ring, sparse=sparse)
    reduced_mat = mat.transpose().rref()
    for row in reduced_mat:
        if row[-1] != 0 and list(row)[:-1] == [0]*(len(row)-1):
            raise Exception("Given function was not a linear combination of basis!")
    for i in range(len(basis)):
        if reduced_mat[i][i] != 1:
            raise Exception("Matrix did not row reduce as expected!")
    return [reduced_mat[i][-1] for i in range(len(basis))]

def polynomial_by_degree(poly, degree_fn = lambda mon: mon.degree()):
    r"""
    Given a polynomial, return a dictionary giving its various parts of fixed degree.

    ``degree_fn`` can be changed to filter the monomials by any degree function.

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: f = x1^2+x2^2+x3+4
        sage: polynomial_by_degree(f) == {2:x1^2+x2^2, 1:x3, 0:4}
        True
        sage: R = generate_multi_polynomial_ring(QQ,2)
        sage: x1,x2,y1,y2 = R.gens()
        sage: bideg = lambda mon: (sum(mon.exponents()[0][:2]),sum(mon.exponents()[0][2:]))
        sage: polynomial_by_degree(x1^2*y2+x2*y1^2, bideg) == {(2,1): x1^2*y2, (1,2): x2*y1^2}
        True
    """
    degrees = set(degree_fn(mon) for mon in poly.monomials())
    return {d:sum(coeff*mon for (coeff, mon) in list(poly) if degree_fn(mon) == d) for d in degrees}

def matrix_to_linear_polynomial_function(mat, domain, codomain, domain_monomial_basis, codomain_basis):
    r"""
    Given a matrix, provide a map on polynomials supported on the monomials in ``domain_monomial_basis`` to polynomials in ``codomain_basis`` where the matrix represents the transition between them.

    EXAMPLES::

        sage: R.<x1,x2> = QQ['x1,x2']
        sage: S.<y1,y2> = QQ['y1,y2']
        sage: mat = matrix(QQ,[[1,0,0],[1,1,0],[1,1,1]])
        sage: dom_mons = monomial_basis(2,R)
        sage: codom_mons = monomial_basis(2,S)
        sage: phi = matrix_to_linear_polynomial_function(mat, R, S, dom_mons, codom_mons)
        sage: phi(2*x1^2+x1*x2)
        2*y1^2 + 3*y1*y2 + 3*y2^2
    """
    return lambda poly: codomain(sum([coeff*cod_bas for (coeff,cod_bas) in zip((mat*polys_to_matrix([poly],base_ring=domain.base_ring(),mons=domain_monomial_basis).transpose()).column(0),codomain_basis)]))

def polynomial_to_coeff_support_list(poly):
    r"""
    Given a polynomial, return a list of tuples of the form (coefficient, monomial exponent vector).

    EXAMPLES::

        sage: A = generate_polynomial_ring(QQ, 3)
        sage: polynomial_to_coeff_support_list(A('x1*x2^2')+4*A('x3'))
        [(1, (1, 2, 0)), (4, (0, 0, 1))]
    """
    coeff_mon_list = list(poly)
    return [(coeff, mon.exponents()[0]) for (coeff, mon) in coeff_mon_list]

def coeff_support_list_to_polynomial(coeff_support_list, poly_ring):
    r"""
    Given a list of tuples of the form (coefficient, monomial exponent vector) along with a target polynomial ring, return the associated polynomial.

    EXAMPLES::

        sage: A = generate_polynomial_ring(QQ, 3)
        sage: coeff_support_list_to_polynomial([(1, (1, 2, 0)), (4, (0, 0, 1))], A)
        x1*x2^2 + 4*x3
    """
    gens = poly_ring.gens()
    return sum(coeff*prod(gens[i]**supp[i] for i in range(len(supp))) for (coeff, supp) in coeff_support_list)

def coefficient_of_monomial_in_polynomial(mon, flat_f):
    r"""
    For a polynomial, give the coefficient of a given monomial, considering other monomial generators as valid coefficients.

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: f = 2*x1^2*x2 + x1*x2^2+x2^3
        sage: coefficient_of_monomial_in_polynomial(x1, f)
        x2^2
        sage: coefficient_of_monomial_in_polynomial(x1^2, f)
        2*x2
        sage: A.<x1,x2,x3> = LaurentPolynomialRing(QQ,'x1,x2,x3')
        sage: g = x1^(-1)*x2 + x1^2*x2^(-3)
        sage: coefficient_of_monomial_in_polynomial(x2^(-3), g)
        x1^2
        sage: coefficient_of_monomial_in_polynomial(x2, g)
        x1^-1
    """
    assert len(mon.exponents()) == 1, "First argument {} is not a monomial!".format(mon)
    mon_exp = mon.exponents()[0]
    gens = parent(flat_f).gens()
    res = 0
    for (coeff, mn) in flat_f:
        mon2_exp = mn.exponents()[0]
        if all(mon_exp[j] == 0 or mon2_exp[j] - mon_exp[j] == 0 for j in range(len(mon_exp))):
            res += coeff*prod(gens[j]**(mon2_exp[j]-(mon_exp[j] if j < len(mon_exp) else 0)) for j in range(len(mon2_exp)))
    return res
