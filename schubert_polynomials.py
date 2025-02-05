# ****************************************************************************
#  Copyright (C) 2024 George H. Seelinger <ghseeli@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
# ***************************************************************************

from sage.all import Permutation, Permutations, QQ, Frac, parent, SchubertPolynomialRing, prod, SymmetricFunctions, Sequence, cached_function, block_matrix, zero_matrix, binomial, IntegerVectors
from sage.rings.polynomial.multi_polynomial_sequence import PolynomialSequence
from functools import reduce
from math import prod
from polynomial_utils import *
from matrix_utils import *
from itertools import permutations
import warnings

Permutations.options(mult='r2l')

def _iterate_operators_from_reduced_word(op_fn, w, poly, alphabet='x'):
    r"""
    For some natural number indexed operators `X_i`, define for permutation ``w`` operator `X_w(f) = X_{i_1} X_{i_2} \cdots X_{i_l}(f)`, where `w = s_{i_1} \cdots s_{i_l}` is any reduced factorization of `w`. Return `X_w(f)`.
    """
    w = Permutation(w)
    res = poly
    red_word = w.reduced_word()
    for i in reversed(red_word):
        res = op_fn(i, res, alphabet=alphabet)
    return res

def s_i(br, i, alphabet='x'):
    r"""
    Return a ring homomorphism swapping variables `x_i` and `x_{i+1}` (or any other letter corresponding to ``alphabet``).

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: (s_i(R,1))(x1-x3)
        x2 - x3
        sage: S.<q,t,x1,x2,x3> = QQ['q,t,x1,x2,x3']
        sage: (s_i(S,2))(x1-x3)
        x1 - x2
    """
    gens_dict = br.gens_dict()
    x = alphabet
    new_dict = dict([(k,v) for (k,v) in gens_dict.items() if k != x+str(i) and k != x+str(i+1)] + [(x+str(i),gens_dict[x+str(i+1)]), (x+str(i+1),gens_dict[x+str(i)])]) 
    return br.hom([new_dict[k] for k in gens_dict.keys()])

def divided_difference_on_monomial_closed(i, mon):
    r"""
    Return the divided difference `\partial_i` on a monomial ``mon``.

    EXAMPLES::

        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: divided_difference_on_monomial_closed(1,x1^5*x2^2)
        x1^4*x2^2 + x1^3*x2^3 + x1^2*x2^4
        sage: divided_difference_on_monomial_closed(1,x1^2*x2^5)
        -x1^4*x2^2 - x1^3*x2^3 - x1^2*x2^4
        sage: divided_difference_on_monomial_closed(1,x1*x2*x3^4)
        0

    """
    supp = list(mon.exponents()[0])
    par = parent(mon)
    xx = par.gens()
    if supp[i] == supp[i-1]:
        return par.zero()
    elif supp[i-1] > supp[i]:
        exp_list = [supp[:i-1]+[supp[i-1]-j,supp[i]-1+j] + supp[i+1:] for j in range(1,abs(supp[i]-supp[i-1])+1)]
    else:
        exp_list = [supp[:i-1]+[supp[i]-j,supp[i-1]-1+j] + supp[i+1:] for j in range(1,abs(supp[i]-supp[i-1])+1)]
    sign = 1 if supp[i-1] > supp[i] else -1
    return par.zero() + sign*sum([prod(xx[j]**expon[j] for j in range(len(expon))) for expon in exp_list]) 

def divided_difference_matrix(i, domain_mons, codomain_mons):
    r"""
    Given a list of monomials, return a matrix whose columns are the images of the monomials, written in the basis given by ``codomain_mons``.

    EXAMPLES::

        sage: R.<x1,x2> = QQ['x1,x2']
        sage: domain_mons = [x1^2, x1*x2, x2^2]
        sage: codomain_mons = [x1, x2]
        sage: divided_difference_matrix(1, domain_mons, codomain_mons)
        [ 1  0  -1]
        [ 1  0  -1]
    """
#    return polys_to_matrix([divided_difference_on_monomial_closed(i,mon) for mon in domain_mons],base_ring=parent(codomain_mons[0]).base_ring(), mons=codomain_mons).transpose()
    if domain_mons and codomain_mons:
        par = parent(domain_mons[0])
        A,v = PolynomialSequence(par,[divided_difference_on_monomial_closed(i,mon) for mon in domain_mons]).coefficients_monomials(order=codomain_mons)
        return A.transpose()
    else:
        warnings.warn("One of domain_mons: " + str(domain_mons) + " or codomain_mons:" + str(codomain_mons) + " is empty, producing an empty matrix!")
        return matrix(QQ,[])

@cached_function
def divided_difference_matrix_x(i, deg, num_vars):
    r"""
    Give a matrix for the operator `\partial_i` applied to the vector space with basis degree ``deg`` monomials in ``num_vars``.

    EXAMPLES::

        sage: divided_difference_matrix_x(1, 2, 2)
        [ 1  0  -1]
        [ 1  0  -1]
        sage: divided_difference_matrix_x(1, 0, 2)
        [0]
    """
    R = generate_polynomial_ring(QQ, num_vars)
    if deg > 0:
        return divided_difference_matrix(i, monomial_basis(deg, R), monomial_basis(deg-1,R))
    elif deg == 0:
        return matrix(QQ, [[0]])
    else:
        raise ValueError("Negative degree divided difference matrix not implemented!")

def divided_difference_matrix_xy(i, xdeg, ydeg, num_vars, lazy=False):
    r"""
    Give a matrix of the operator `\partial_i^{(x)}` applied to the vector space with basis of monomials in ``num_vars`` `x`-variables and ``num_vars`` `y`-variables with `x`-degree ``xdeg`` and `y`-degree ``ydeg``.

    Note, this is a degree (-1,0)-operator, so the `y`-degree of the result stays the same.

    EXAMPLES::

        sage: R = generate_multi_polynomial_ring(QQ, 2)
        sage: divided_difference_matrix_xy(1,2,2,2) == divided_difference_matrix(1,monomial_basis_in_fixed_xy_degree(2,2,R),monomial_basis_in_fixed_xy_degree(1,2,R))
        True
        sage: divided_difference_matrix_xy(1,2,1,3,True).to_matrix() == divided_difference_matrix_xy(1,2,1,3,False)
        True
    """
    y_dim = len(IntegerVectors(ydeg, num_vars))
    dd_mat_x = divided_difference_matrix_x(i, xdeg, num_vars)
    m = dd_mat_x.nrows()
    n = dd_mat_x.ncols()
    if not lazy:
        return tensor_product_of_sparse_rational_matrices(dd_mat_x, matrix.identity(y_dim,sparse=True))
    else:
        return LazyBlockMatrix([[dd_mat_x[i][j]*matrix.identity(y_dim,sparse=True) for j in range(n)] for i in range(m)])
        #return LazyBlockMatrix([[matrix(m,n,{})]*i + [dd_mat_x] + [matrix(m,n,{})]*(y_dim-i-1) for i in range(y_dim)])
    
    #return divided_difference_matrix_x(i, xdeg, num_vars).tensor_product(matrix.identity(y_dim), subdivide=False)

def divided_difference(i, poly, alphabet='x'):
    r"""
    Return the divided difference `\partial_i` on ``poly``, computed directly from the definition.

    EXAMPLES::
    
        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: divided_difference(1,x1+x2)
        0
        sage: divided_difference(1,x1+x3) == (x1-x2)/(x1-x2)
        True
        sage: divided_difference(2,x1^2*x2) == (x1^2*x2-x1^2*x3)/(x2-x3)
        True
    """
    br = parent(poly)
    x = alphabet
    si = s_i(br, i, x)
    return 1/(br(x+str(i)) - br(x+str(i+1)))*(poly-si(poly))

    
def divided_difference_w(w, poly, alphabet='x'):
    r"""
    Return the composition of divided difference operators corresponding to a reduced
    word of permutation `w`.

    EXAMPLES::

        sage: R.<x1,x2,x3> = Frac(QQ['x1,x2,x3'])
        sage: poly = x1^2*x2
        sage: divided_difference_w([3,2,1], poly) 
        1
        sage: divided_difference_w([3,2,1], poly) == divided_difference(1, divided_difference(2, divided_difference(1, poly)))
        True
        sage: divided_difference_w([3,2,1], poly) == divided_difference(2, divided_difference(1, divided_difference(2, poly)))
        True
        sage: divided_difference_w([2,3,1], poly) == x1+x2
        True
        sage: divided_difference_w([3,1,2], poly) == x1
        True
    """
    return _iterate_operators_from_reduced_word(divided_difference, w, poly, alphabet=alphabet)

def divided_difference_w_via_matrix_homogeneous(w, poly, xdeg, ydeg, num_vars, lazy=False):
    r"""
    Apply the divided difference operator `\partial_w` to `poly` of fixed `x`-degree and `y`-degree. 

    EXAMPLES::

        sage: R = generate_multi_polynomial_ring(QQ,3)
        sage: poly = R('x1^2*y1-x1*x2*y1-x1*x2*y2')
        sage: divided_difference_w_via_matrix_homogeneous([2,1,3],poly,2,1,3) == divided_difference_w([2,1,3],poly)
        True
        sage: divided_difference_w_via_matrix_homogeneous([2,1,3],poly,2,1,3,True) == divided_difference_w([2,1,3],poly)
        True
    """
    word = list(reversed(Permutation(w).reduced_word()))
    if xdeg < len(word):
        return parent(poly).zero()
    mats = [divided_difference_matrix_xy(word[i], xdeg-i, ydeg, num_vars, lazy) for i in range(len(word))]
    res_vec = polys_to_matrix([poly], base_ring=parent(poly).base_ring(), mons=monomial_basis_in_fixed_xy_degree(xdeg, ydeg, parent(poly))).row(0)#.transpose().column(0)
    for mat in mats:
        if not lazy:
            res_vec = mat*res_vec
        else:
            res_vec = mat.apply_to_vector(res_vec)
    return sum([coeff*mon for (coeff,mon) in zip(res_vec,monomial_basis_in_fixed_xy_degree(xdeg-len(word), ydeg, parent(poly)))])


def _polynomial_by_xy_bidegree(poly, num_x_vars):
    r"""
    Return a dictionary giving the `xy`-bidegree homogeneous pieces of ``poly``, assuming ``poly`` is a polynomial in `x_1,\ldots,x_n,y_1,\ldots,y_m` for `n` equal to ``num_x_vars``.

    Note, this will return incorrect results if there are other algebraic generators!
    This function is primarily meant for computations of double Schubert polynomials and related families and is not meant as a general purpose function.
    See ``polynomial_by_degree`` in ``polynomial_utils.py`` for a general function.
    """
    bideg = lambda mon: (sum(mon.exponents()[0][:num_x_vars]),sum(mon.exponents()[0][num_x_vars:]))
    return polynomial_by_degree(poly, bideg)

def divided_difference_w_via_matrix(w, poly, lazy=False):
    num_vars = len(w)
    return  sum(divided_difference_w_via_matrix_homogeneous(w,homog_poly,xdeg,ydeg,num_vars,lazy) for ((xdeg, ydeg),homog_poly) in _polynomial_by_xy_bidegree(poly,num_vars).items())

def double_schubert_poly(w, direct=True):
    r"""
    Return the double Schubert polynomial corresponding to permutation `w`.

    If ``direct`` is ``True``, the polynomial will be computed directly using the definition of divided difference operators.
    If ``direct`` is ``False``, it will first compute the matrix of each relevant divided difference operator and use the matrices to compute the polynomials.

    EXAMPLES::
    
        sage: R.<x1,x2,x3,y1,y2,y3> = QQ['x1,x2,x3,y1,y2,y3']
        sage: double_schubert_poly([3,2,1]) == ((x1-y1)*(x1-y2)*(x2-y1))
        True
        sage: double_schubert_poly([1,3,2]) == x1 + x2 - y1 - y2
        True
        sage: double_schubert_poly([1,3,2], direct=False) == x1 + x2 - y1 - y2
        True
    """
    w = Permutation(w)
    n = len(w)
    br = Frac(generate_multi_polynomial_ring(QQ, n))
    base_poly = prod([br('x'+str(i+1))-br('y'+str(j+1)) for i in range(n) for j in range(n) if i+j+2 <= n])
    longest_word = Permutation(range(n,0,-1))
    if direct:
        return divided_difference_w(w.inverse()*longest_word, base_poly).numerator()
    else:
        return divided_difference_w_via_matrix(w.inverse()*longest_word, base_poly.numerator(),False)

## Grothendieck polynomials

def pi_divided_difference(i, poly, alphabet='x'):
    r"""
    Apply the operator `\pi_i(f) = \partial_i((1-x_{i+1})f)` to ``poly``.
    """
    br = parent(poly)
    x = alphabet
    return divided_difference(i, (1-br(x+str(i+1)))*poly, alphabet=x)

def _dd_matrix_on_monomials_of_bounded_x_degree_and_fixed_y_degree(i, max_x_deg, y_deg, num_vars):
    r"""
    Return a square matrix representing the action of a divided difference operator on monomials of bounded `x`-degree and fixed `y`-degree.

    EXAMPLES::

        sage: _dd_matrix_on_monomials_of_bounded_x_degree_and_fixed_y_degree(1, 2, 0, 2)
        [ 0  1 -1  0  0  0]
        [ 0  0  0  1  0 -1]
        [ 0  0  0  1  0 -1]
        [ 0  0  0  0  0  0]
        [ 0  0  0  0  0  0]
        [ 0  0  0  0  0  0]
        sage: _dd_matrix_on_monomials_of_bounded_x_degree_and_fixed_y_degree(1, 1, 1, 2)
        [ 0  0  1  0 -1  0]
        [ 0  0  0  1  0 -1]
        [ 0  0  0  0  0  0]
        [ 0  0  0  0  0  0]
        [ 0  0  0  0  0  0]
        [ 0  0  0  0  0  0]
    """
    m = max_x_deg+1
    dd_matrices = [divided_difference_matrix_xy(i, xdeg, y_deg, num_vars) for xdeg in range(1,m)]
    dd_block_matrix_top = block_matrix([[zero_matrix(QQ,dd_matrices[i].nrows(),len(IntegerVectors(y_deg,num_vars)),sparse=True)] + [0]*(i) + [dd_matrices[i]] + [0]*(m-2-i) for i in range(m-1)],sparse=True)
    dd_block_matrix_bottom = zero_matrix(QQ,dd_block_matrix_top.ncols()-dd_block_matrix_top.nrows(),dd_block_matrix_top.ncols(),sparse=True)
    dd_block_matrix = block_matrix([[dd_block_matrix_top],[dd_block_matrix_bottom]],subdivide=False,sparse=True)
    return dd_block_matrix

def pi_divided_difference_matrix_xy(i, max_x_deg, y_deg, num_vars):
    r"""
    Return the matrix of the divided difference operator `\pi_i` on the space of monomials bounded in `x`-degree and with fixed `y`-degree.

    EXAMPLES::

        sage: R.<x1,x2> = QQ['x1,x2']
        sage: pi_divided_difference_matrix_xy(1, 2, 0, 2)
        [ 1  1 -1  0  0  0]
        [ 0  0  1  1  0 -1]
        [ 0  0  1  1  0 -1]
        [ 0  0  0  0  0  1]
        [ 0  0  0 -1  1  1]
        [ 0  0  0  0  0  1]

    """
    br = generate_multi_polynomial_ring(QQ, num_vars)
    domain_mons = reduce(lambda a,b: a+b, [monomial_basis_in_fixed_xy_degree(m, y_deg, br) for m in range(max_x_deg+1)])
    codomain_mons = domain_mons + monomial_basis_in_fixed_xy_degree(max_x_deg+1, y_deg, br)
    mult_by_one_minus_xip1,v = PolynomialSequence([(1-br('x'+str(i+1)))*mon for mon in domain_mons]).coefficients_monomials(order=codomain_mons)
    mult_by_one_minus_xip1_op = mult_by_one_minus_xip1.transpose()
    dd_matrix = _dd_matrix_on_monomials_of_bounded_x_degree_and_fixed_y_degree(i, max_x_deg+1, y_deg, num_vars)
    return (dd_matrix*mult_by_one_minus_xip1_op)[:len(domain_mons)]

def pi_divided_difference_w_via_matrix_y_homogeneous(w, poly, max_x_deg, y_deg, num_vars):
    r"""
    Return the result of `\pi_w` on ``poly`` of fixed `y`-degree and of max `x`-degree ``max_x_deg`` using matrix methods. 

    EXAMPLES::

        sage: R.<x1,x2,x3,y1,y2,y3> = QQ['x1,x2,x3,y1,y2,y3']
        sage: pi_divided_difference_w_via_matrix_y_homogeneous([3,1,2],x2^2*y1, 2, 1, 3)
        x1^2*y1 - x2*x3*y1 + x2*y1 + x3*y1 - y1
    """
    word = list(reversed(Permutation(w).reduced_word()))
    mats = [pi_divided_difference_matrix_xy(word[i], max_x_deg, y_deg, num_vars) for i in range(len(word))]
    mons = reduce(lambda a,b: a+b, [monomial_basis_in_fixed_xy_degree(m, y_deg, parent(poly)) for m in range(max_x_deg+1)])
    res_vec = polys_to_matrix([poly], base_ring=parent(poly).base_ring(), mons=mons).transpose()
    for mat in mats:
        res_vec = mat*res_vec
    return sum([coeff*mon for (coeff,mon) in zip(res_vec.column(0),mons)])

def pi_divided_difference_w(w, poly, alphabet='x'):
    r"""
    Apply the operator `\pi_w = \pi_{s_{i_1}} \cdots \pi_{s_{i_l}}` for any reduced factorization of `w = s_{i_1} \cdots s_{i_l}`.

    Note, ``w`` should be a permutation given in 1-line notation.
    """
    return _iterate_operators_from_reduced_word(pi_divided_difference, w, poly, alphabet=alphabet)

def _polynomial_by_y_degree(poly, num_x_vars):
    ydeg = lambda mon: sum(mon.exponents()[0][num_x_vars:])
    return polynomial_by_degree(poly, ydeg)

def pi_divided_difference_w_via_matrix(w, poly):
    r"""
    Return the result of `\pi_w` divided difference on ``poly`` using matrix methods.

    EXAMPLES::

        sage: R.<x1,x2,x3,y1,y2,y3> = QQ['x1,x2,x3,y1,y2,y3']
        sage: pi_divided_difference_w_via_matrix([3,1,2],x1^2*x2*y1^2*y2+x1^2*y1^2) == pi_divided_difference_w([3,1,2],x1^2*x2*y1^2*y2+x1^2*y1^2)
        True
    """
    num_vars = len(w)
    return sum(pi_divided_difference_w_via_matrix_y_homogeneous(w, homog_poly, homog_poly.degree()-ydeg, ydeg, num_vars) for (ydeg, homog_poly) in _polynomial_by_y_degree(poly, num_vars).items())

def grothendieck_poly(w, x_pref='x', direct=True):
    r"""
    Return the Grothendieck polynomial associated with permutation ``w`` in monomials. 

    EXAMPLES::

        sage: grothendieck_poly([1,3,2])
        -x1*x2 + x1 + x2
        sage: grothendieck_poly([2,3,1])
        x1*x2
        sage: grothendieck_poly([1,4,2,3])
        -x1^2*x2 - x1*x2^2 + x1^2 + x1*x2 + x2^2
        sage: grothendieck_poly([1,4,2,3], direct=False)
        -x1^2*x2 - x1*x2^2 + x1^2 + x1*x2 + x2^2
    """
    n = len(w)
    w = Permutation(w)
    poly_ring = generate_polynomial_ring(QQ, n, x_pref=x_pref)
    br = Frac(poly_ring)
    base_poly = prod([br(x_pref+str(i+1))**(n-i-1) for i in range(n)])
    longest_word = Permutation(range(n,0,-1))
    if direct:
        return poly_ring.one()*poly_ring(pi_divided_difference_w(w.inverse()*longest_word, base_poly))
    else:
        return poly_ring.one()*poly_ring(pi_divided_difference_w_via_matrix(w.inverse()*longest_word, base_poly.numerator()))

def double_grothendieck_poly(w, direct=True):
    r"""
    Return the double Grothendieck polynomial associated with permutation ``w`` in monomials. 

    EXAMPLES::

        sage: double_grothendieck_poly([1,3,2])
        -x1*x2*y1*y2 + x1*x2*y1 + x1*x2*y2 + x1*y1*y2 + x2*y1*y2 - x1*x2 - x1*y1 - x2*y1 - x1*y2 - x2*y2 - y1*y2 + x1 + x2 + y1 + y2
        sage: double_grothendieck_poly([3,1,2])
        x1^2*y1*y2 - x1^2*y1 - x1^2*y2 - 2*x1*y1*y2 + x1^2 + x1*y1 + x1*y2 + y1*y2
        sage: double_grothendieck_poly([3,1,2], direct=False) == double_grothendieck_poly([3,1,2])
        True
    """
    w = Permutation(w)
    n = len(w)
    poly_ring = generate_multi_polynomial_ring(QQ, n)
    br = Frac(poly_ring)
    base_poly = prod([br('x'+str(i+1))+br('y'+str(j+1))-br('x'+str(i+1))*br('y'+str(j+1)) for i in range(n) for j in range(n) if i+j+2 <= n])
    longest_word = Permutation(range(n,0,-1))
    if direct:
        return poly_ring(pi_divided_difference_w(w.inverse()*longest_word, base_poly))
    else:
        return pi_divided_difference_w_via_matrix(w.inverse()*longest_word, base_poly.numerator())

## Quantum Schubert polynomials

def e_i_m(i, m, ambient_vars, br=QQ, start=1):
    r"""
    Return the elementary symmetric function `e_i(x_1,\ldots,x_m)`.
    """
    var_string = reduce(lambda a,b: a+','+b, ['x'+str(j) for j in range(start,start+ambient_vars)])
    ambient_ring = br[var_string]
    sym = SymmetricFunctions(br)
    e = sym.e()
    zero_vars = {ambient_ring('x'+str(j)):0 for j in range(start+m,start+ambient_vars)}
    return e[i].expand(ambient_vars,var_string).subs(zero_vars)

def e_basis_elm(seq, br=QQ, start=1):
    r"""
    For ``seq`` a sequence of integers `(a_1,\ldots,a_m)`, return `e_{a_1}(x_1) e_{a_2}(x_1,x_2) \cdots e_{a_m}(x_1,\ldots,x_m)`.
    """
    m = len(seq)
    return prod([e_i_m(seq[i],i+1,m,br,start) for i in range(m)])

def e_basis(deg, l, br=QQ, start=1):
    r"""
    For a fixed degree ``deg`` and number of variables ``l``, return the basis of degree ``deg`` polynomials in ``l`` variables given by products of elementary symmetric functions in increasing numbers of variables.
    """
    elms = [e_basis_elm(seq,br,start) for seq in IntegerVectors(deg,length=l,outer=list(range(1,l+1)))]
    return [elm for elm in elms if elm != 0]

def Schubert_in_e(perm, base_ring=QQ, zeroes=True):
    r"""
    Return the expansion of the Schubert polynomial indexed by ``perm``, a permutation of `n`, in the elementary symmetric function product basis of polynomials in ``n`` variables.

    EXAMPLES::

        sage: Schubert_in_e([2,1,3])
        [(1, [1, 0]), (0, [0, 1])]
        sage: Schubert_in_e([2,1,3], zeroes=False)
        [(1, [1, 0])]
        sage: Schubert_in_e([3,2,1])
        [(1, [1, 2])]
    """
    l = len(perm)
    X = SchubertPolynomialRing(base_ring)
    poly = X(perm).expand() # Note, Sage gives 0-indexed Schubert polynomials
    d = poly.degree()
    coeffs_in_e = solve_polynomial_in_terms_of_basis(poly, e_basis(d, l-1, br=base_ring, start=0), base_ring)
    res = list(zip(coeffs_in_e,IntegerVectors(d,length=l-1,outer=list(range(1,l+1))))) 
    if zeroes:
        return res
    else:
        return [(coeff, supp) for (coeff, supp) in res if coeff != 0]

def generate_quantum_polynomial_ring(br, num_vars, x_pref='x', start=1):
    r"""
    Return the polynomial ring with ``num_vars`` x-variables and ``num_vars`` q-variables.
    """
    xq_vars = ['x'+str(j+start) for j in range(num_vars)]+['q'+str(j+start) for j in range(num_vars)] 
    return PolynomialRing(br, xq_vars, 2*num_vars)

def quantum_e_i_m(i, m, ambient_vars, br=QQ, start=1):
    r"""
    Return the quantum ``E_i^m`` polynomial.
    """
    ambient_ring = generate_quantum_polynomial_ring(br, ambient_vars, start=start)
    if i < 0 or i > m:
        return ambient_ring.zero()
    if m < 0:
        return ambient_ring.zero()
    if i == 0:
        return ambient_ring.one()
    if m == 1:
        return quantum_e_i_m(i,m-1,ambient_vars,br,start)+ambient_ring('x'+str(m-1+start))*quantum_e_i_m(i-1,m-1,ambient_vars,br,start)
    return quantum_e_i_m(i,m-1,ambient_vars,br,start)+ambient_ring('x'+str(m-1+start))*quantum_e_i_m(i-1,m-1,ambient_vars,br,start) + quantum_e_i_m(i-2,m-2,ambient_vars,br,start)*ambient_ring('q'+str(m-2+start))

def quantum_e_seq(seq, br=QQ, start=1):
    m = len(seq)
    return prod([quantum_e_i_m(seq[i],i+1,m,br,start) for i in range(m)]) 

def quantum_e_basis(deg, l, br=QQ, start=1):
    elms = [quantum_e_seq(seq,br,start) for seq in IntegerVectors(deg,length=l)]
    return [elm for elm in elms if elm != 0]

def quantum_Schubert(perm, base_ring=QQ, start=1):
    r"""
    Return the quantum Schubert polynomial associated to permutation ``perm`` from the corresponding paper by Fomin-Gelfand-Postnikov.

    EXAMPLES::

        sage: quantum_Schubert([3,1,2])
        x1^2 - q1
        sage: quantum_Schubert([1,2,3])
        1
        sage: quantum_Schubert([1,3,2])
        x1 + x2
        sage: A = generate_quantum_polynomial_ring(QQ, 3)
        sage: x1,x2,x3,q1,q2,q3 = A.gens()
        sage: quantum_Schubert([3,4,1,2]) == x1*x1*x2*x2 + 2*q1*x1*x2 - q2*x1*x1 + q1*q1 + q1*q2
        True 
    """
    schub_in_e = Schubert_in_e(perm, base_ring)
    coeffs_in_e = [coeff for (coeff,supp) in schub_in_e]
    d = Permutation(perm).length()
    poly_in_quantum_E = sum([coeff*elm for (elm,coeff) in zip(quantum_e_basis(d, len(perm)-1, br=base_ring, start=start),coeffs_in_e)])
    return poly_in_quantum_E

def generate_quantum_Schubert_basis(br, num_vars):
    r"""
    Returns dictionary with quantum Schubert polynomial as key and its corresponding permutation as value in given base ring.
    
    EXAMPLES::
        sage: A = Frac(QQ['q1,q2'])['x1,x2,x3']
        sage: generate_quantum_Schubert_basis(A, 3)
        {1: [1, 2, 3],
         x1: [2, 1, 3],
         x1 + x2: [1, 3, 2],
         x1*x2 + q1: [2, 3, 1],
         x1^2 + (-q1): [3, 1, 2],
         x1^2*x2 + q1*x1: [3, 2, 1]}

    USE CASE (in above example)::
        sage: qs_polys = list(generate_quantum_Schubert_basis(A, 3).keys())
        sage: solve_polynomial_in_terms_of_basis(A(quantum_Schubert([2,1,3]) * quantum_Schubert([2,1,3])), qs_polys, A.base_ring())
        [q1, 0, 0, 0, 1, 0]
    """
    quantum_Schubert_dict = {}
    
    for perm in list(permutations([i for i in range(1, num_vars + 1)])):
        quantum_Schubert_dict[br(quantum_Schubert(list(perm)))] = list(perm)
    return quantum_Schubert_dict
    
## Quantum Grothendieck polynomials

def quantum_F_p_k(p, k, ambient_vars, br=QQ, x_pref='x', start=1):
    r"""
    Return the polynomial `F_p^k` from Equation (3.1) in Lenart-Maeno.
    """
    assert 0 <= p
    assert k <= ambient_vars
    ambient_ring = generate_quantum_polynomial_ring(br, ambient_vars, x_pref, start)
    if p > k:
        return ambient_ring.zero()
    if p==0:
        return ambient_ring.one()
    Is = Subsets(range(start,start+k),p)
    return ambient_ring.zero() + sum(prod(1-ambient_ring(x_pref+str(i)) for i in I)*prod(1-ambient_ring('q'+str(i)) for i in I if i+1 not in I) for I in Is)

def quantum_F_p_k_bar(p, k, ambient_vars, br=QQ, start=1):
    r"""
    Return the `\overline{F}_p^k` polynomial from Lenart-Maeno, given by specializing `q_k = 0` in `F_p^k`.
    """
    F_p_k = quantum_F_p_k(p,k,ambient_vars,br=br,start=1)
    par = parent(F_p_k)
    return F_p_k.subs({par('q'+str(k-1+start)):par.zero()})

def quantum_F_p_k_tilde(p, k, ambient_vars, br=QQ, start=1):
    r"""
    Compute the `\tilde{F}_p^k` polynomial from equation (3.2) in Lenart-Maeno.
    """
    assert 0 <= p
    assert p <= k
    assert k <= ambient_vars
    ambient_ring = generate_quantum_polynomial_ring(br, ambient_vars, start=start)
    if p==0:
        return ambient_ring.one()
    Is = Subsets(range(start,start+k),p)
    return ambient_ring.zero() + sum(prod(ambient_ring('x'+str(i)) for i in I)*prod(1-ambient_ring('q'+str(i-1)) for i in I-Set([0]) if i-1 not in I) for I in Is)

def quantum_E_p_k_hat(p,k,ambient_vars,br=QQ,start=1):
    assert p >= 0
    return sum((-1)**(i)*binomial(k-i,p-i)*quantum_F_p_k(i,k,ambient_vars,br=br,start=start) for i in range(p+1))

def quantum_E_hat_seq(seq, br=QQ, start=1):
    m = len(seq)
    return prod(quantum_E_p_k_hat(seq[i],i+1,m,br,start) for i in range(m))

def quantum_E_hat_basis(deg, l, br=QQ, start=1):
    elms = [quantum_E_hat_seq(seq, br, start) for d in reversed(range(deg+1)) for seq in IntegerVectors(d,length=l)]
    return [elm for elm in elms if elm != 0]

def inhomog_e_basis(top_deg, l, br=QQ, start=1):
    return [elm for d in reversed(range(top_deg+1)) for elm in e_basis(d, l, br=br, start=start)]

def Grothendieck_in_e(perm, base_ring=QQ, zeroes=True):
    r"""
    Return the expansion of the Grothendieck polynomial indexed by ``perm``, a permutation of `n`, in the elementary symmetric function product basis of polynomial in ``n`` variables.

    EXAMPLES::

        sage: Grothendieck_in_e([1,3,2])
        [(0, [2, 0]), (-1, [1, 1]), (0, [0, 2]), (1, [1, 0]), (0, [0, 1])]
    """
    poly = grothendieck_poly(perm)
    d = poly.degree()
    l = len(perm)
    basis = inhomog_e_basis(d, len(perm)-1, br=base_ring)
    coeffs_in_e = solve_polynomial_in_terms_of_basis(poly, basis, base_ring)
    indexing_set = [vec for de in reversed(range(d+1)) for vec in IntegerVectors(de,l-1,outer=list(range(1,l+1)))]
    res = list(zip(coeffs_in_e,indexing_set))
    if zeroes:
        return res
    else:
        return [(coeff, supp) for (coeff, supp) in res if coeff != 0]

def quantum_Grothendieck(perm, base_ring=QQ):
    r"""
    Compute the quantum Grothendieck polynomial associated to ``perm`` as in Lenart-Maeno.

    EXAMPLES::

        sage: quantum_Grothendieck([1,2,3])
        1
        sage: quantum_Grothendieck([2,1,3])
        -x1*q1 + x1 + q1
        sage: quantum_Grothendieck([3,2,1]) == quantum_Grothendieck([2,1,3])*quantum_Grothendieck([2,3,1])
        True
        sage: A = generate_quantum_polynomial_ring(QQ, 3)
        sage: x1,x2,x3,q1,q2,q3 = A.gens()
        sage: quantum_Grothendieck([2,3,1]) == (1-q2)*x1*x2-(q1-q2)*x1+q1
        True
    """
    l = len(perm)
    poly = grothendieck_poly(perm)
    poly_par = parent(poly)
    if poly == poly_par.one():
        return quantum_E_hat_seq([0]*l)
    d = poly.degree()
    basis = inhomog_e_basis(d, len(perm)-1, br=base_ring)
    coeffs_in_e = solve_polynomial_in_terms_of_basis(poly, basis, base_ring)
    poly_in_quantum_E = sum([coeff*elm for (elm,coeff) in zip(quantum_E_hat_basis(d, len(perm)-1, br=base_ring),coeffs_in_e)])
    return poly_in_quantum_E
