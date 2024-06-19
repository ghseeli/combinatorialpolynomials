# ****************************************************************************
#  Copyright (C) 2024 George H. Seelinger <ghseeli@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
# ***************************************************************************

from sage.all import Permutation, Permutations, QQ, Frac, parent, SchubertPolynomialRing, prod, SymmetricFunctions
from functools import reduce
from math import prod
from polynomial_utils import *
from itertools import permutations

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

def divided_difference(i, poly, alphabet='x'):
    r"""
    Return the divided difference `\partial_i` on ``poly``.

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

def double_schubert_poly(w):
    r"""
    Return the double Schubert polynomial corresponding to permutation `w`.

    EXAMPLES::
    
        sage: R.<x1,x2,x3,y1,y2,y3> = Frac(QQ['x1,x2,x3,y1,y2,y3'])
        sage: double_schubert_poly([3,2,1]) == (x1-y1)*(x1-y2)*(x2-y1)
        True

    """
    w = Permutation(w)
    n = len(w)
    br = Frac(generate_multi_polynomial_ring(QQ, n))
    base_poly = prod([br('x'+str(i+1))-br('y'+str(j+1)) for i in range(n) for j in range(n) if i+j+2 <= n])
    longest_word = Permutation(range(n,0,-1))
    return divided_difference_w(w.inverse()*longest_word, base_poly)

## Grothendieck polynomials

def pi_divided_difference(i, poly, alphabet='x'):
    r"""
    Apply the operator `\pi_i(f) = \partial_i((1-x_{i+1})f)` to ``poly``.
    """
    br = parent(poly)
    x = alphabet
    return divided_difference(i, (1-br(x+str(i+1)))*poly, alphabet=x)

def pi_divided_difference_w(w, poly, alphabet='x'):
    r"""
    Apply the operator `\pi_w = \pi_{s_{i_1}} \cdots \pi_{s_{i_l}}` for any reduced factorization of `w = s_{i_1} \cdots s_{i_l}`.

    Note, ``w`` should be a permutation given in 1-line notation.
    """
    return _iterate_operators_from_reduced_word(pi_divided_difference, w, poly, alphabet=alphabet)


def grothendieck_poly(w, x_pref='x'):
    r"""
    Return the Grothendieck polynomial associated with permutation ``w`` in monomials. 

    EXAMPLES::

        sage: grothendieck_poly([1,3,2])
        -x1*x2 + x1 + x2
        sage: grothendieck_poly([2,3,1])
        x1*x2
        sage: grothendieck_poly([1, 4, 2, 3])
        -x1^2*x2 - x1*x2^2 + x1^2 + x1*x2 + x2^2
    """
    n = len(w)
    w = Permutation(w)
    poly_ring = generate_polynomial_ring(QQ, n, x_pref=x_pref)
    br = Frac(poly_ring)
    base_poly = prod([br(x_pref+str(i+1))**(n-i-1) for i in range(n)])
    longest_word = Permutation(range(n,0,-1))
    return poly_ring.one()*poly_ring(pi_divided_difference_w(w.inverse()*longest_word, base_poly))

def double_grothendieck_poly(w):
    r"""
    Return the double Grothendieck polynomial associated with permutation ``w`` in monomials. 

    EXAMPLES::

        sage: double_grothendieck_poly([1,3,2])
        -x1*x2*y1*y2 + x1*x2*y1 + x1*x2*y2 + x1*y1*y2 + x2*y1*y2 - x1*x2 - x1*y1 - x2*y1 - x1*y2 - x2*y2 - y1*y2 + x1 + x2 + y1 + y2
    """
    w = Permutation(w)
    n = len(w)
    poly_ring = generate_multi_polynomial_ring(QQ, n)
    br = Frac(poly_ring)
    base_poly = prod([br('x'+str(i+1))+br('y'+str(j+1))-br('x'+str(i+1))*br('y'+str(j+1)) for i in range(n) for j in range(n) if i+j+2 <= n])
    longest_word = Permutation(range(n,0,-1))
    return poly_ring(pi_divided_difference_w(w.inverse()*longest_word, base_poly))

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
    elms = [e_basis_elm(seq,br,start) for seq in IntegerVectors(deg,length=l)]
    return [elm for elm in elms if elm != 0]

def Schubert_in_e(perm, base_ring=QQ):
    r"""
    Return the expansion of the Schubert polynomial indexed by ``perm``, a permutation of `n`, in the elementary symmetric function product basis of polynomials in ``n`` variables.

    EXAMPLES::

        sage: Schubert_in_e([2,1,3])
        [(1, [1, 0]), (0, [0, 1])]
    """
    l = len(perm)
    X = SchubertPolynomialRing(base_ring)
    poly = X(perm).expand() # Note, Sage gives 0-indexed Schubert polynomials
    d = poly.degree()
    coeffs_in_e = solve_polynomial_in_terms_of_basis(poly, e_basis(d, len(perm)-1, br=base_ring, start=0), base_ring)
    return list(zip(coeffs_in_e,IntegerVectors(d,length=l-1)))

def generate_quantum_polynomial_ring(br, num_vars, x_pref='x', start=1):
    r"""
    Return the polynomial ring with ``num_vars`` x-variables and ``num_vars`` q-variables.
    """
    xq_vars = ['x'+str(j+start) for j in range(num_vars)]+['q'+str(j+start) for j in range(num_vars)] 
    return PolynomialRing(br, xq_vars, 2*num_vars)

def quantum_e_i_m(i,m,ambient_vars,br=QQ, start=1):
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
        {1: [1, 2, 3], x1 + x2: [1, 3, 2], x1: [2, 1, 3], x1*x2 + q1: [2, 3, 1], x1^2 + (-q1): [3, 1, 2], x1^2*x2 + q1*x1: [3, 2, 1]}

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
