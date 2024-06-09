# ****************************************************************************
#  Copyright (C) 2024 George H. Seelinger <ghseeli@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
# ***************************************************************************

from sage.all import Permutation, QQ, Frac, parent
from polynomial_utils import *

Permutations.options(mult='r2l')

def _iterate_operators_from_reduced_word(op_fn, w, poly, alphabet='x'):
    w = Permutation(w)
    res = poly
    red_word = w.reduced_word()
    for i in reversed(red_word):
        res = op_fn(i, res, alphabet=alphabet)
    return res

def s_i(br, i, alphabet='x'):
    r"""
    Return a ring homomorphism swapping variables ``xi`` and ``xi+1``.

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
    n = len(w)
    br = Frac(generate_multi_polynomial_ring(QQ, n))
    base_poly = prod([br('x'+str(i+1))-br('y'+str(j+1)) for i in range(n) for j in range(n) if i+j+2 <= n])
    longest_word = Permutation(range(n,0,-1))
    return divided_difference_w(w.inverse()*longest_word, base_poly)

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
    """
    n = len(w)
    poly_ring = generate_polynomial_ring(QQ, num_vars, x_pref=x_pref)
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
    n = len(w)
    poly_ring = generate_multi_polynomial_ring(QQ, n)
    br = Frac(poly_ring)
    base_poly = prod([br('x'+str(i+1))+br('y'+str(j+1))-br('x'+str(i+1))*br('y'+str(j+1)) for i in range(n) for j in range(n) if i+j+2 <= n])
    longest_word = Permutation(range(n,0,-1))
    return poly_ring(pi_divided_difference_w(w.inverse()*longest_word, base_poly))

def e_i_m(i, m, ambient_vars, br=QQ, start=1):
    var_string = reduce(lambda a,b: a+','+b, ['x'+str(j) for j in range(start,start+ambient_vars)])
    ambient_ring = br[var_string]
    sym = SymmetricFunctions(br)
    e = sym.e()
    zero_vars = {ambient_ring('x'+str(j)):0 for j in range(start+m,start+ambient_vars)}
    return e[i].expand(ambient_vars,var_string).subs(zero_vars)

def e_basis_elm(seq, br=QQ, start=1):
    m = len(seq)
    return prod([e_i_m(seq[i],i+1,m,br,start) for i in range(m)])

def e_basis(deg, l, br=QQ, start=1):
    elms = [e_basis_elm(seq,br,start) for seq in IntegerVectors(deg,length=l)]
    return [elm for elm in elms if elm != 0]

def Schubert_in_e(perm, base_ring=QQ):
    l = len(perm)
    X = SchubertPolynomialRing(base_ring)
    poly = X(perm).expand() # Note, Sage gives 0-indexed Schubert polynomials
    d = poly.degree()
    coeffs_in_e = solve_function_in_terms_of_basis(poly, e_basis(d, len(perm)-1, br=base_ring, start=0), base_ring)
    return list(zip(coeffs_in_e,IntegerVectors(d,length=l-1)))

def generate_quantum_polynomial_ring(br, num_vars, x_pref='x', start=1):
    r"""
    Return the polynomial ring with ``num_vars`` x-variables and ``num_vars``-1 q-variables.
    """
    xq_vars = ['x'+str(j+start) for j in range(num_vars)]+['q'+str(j+start) for j in range(num_vars-1)] 
    return PolynomialRing(br, xq_vars, 2*num_vars-1)

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

