# ***************************************************************************
#
#  Copyright (C) 2025 George H. Seelinger <ghseeli@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
# ***************************************************************************

from sage.all import cartesian_product, MixedIntegerLinearProgram, parent, Permutation, Permutations, prod, QQ, Rationals, RootSystem, Word
from polynomial_utils import generate_laurent_polynomial_ring, invert_variables_in_flat_polynomial, separate_polynomial_generators, specialize_flat_polynomial_variables
from schubert_polynomials import s_i, s_i_on_polynomial, divided_difference, divided_difference_on_polynomial, _iterate_operators_from_reduced_word

def demazure_pi_i(i, f, alphabet='x'):
    r"""
    Apply the (isobaric) Demazure `\pi_i` operator to a multivariate function.

    Note, `\pi_i(f) = \frac{x_i f - x_{i+1}s_i(f)}{x_i-x_{i+1}} = \partial_i(x_i f)` for `\partial_i` the divided difference operator.

    Equivalently, `\pi_i(f) = (1+s_i) \frac{f}{1-x_{i+1}/x_i}` for a more Lie-theoretic definition.

    EXAMPLES::

        sage: A.<x1,x2> = QQ['x1,x2']
        sage: demazure_pi_i(1, x1^2*x2)
        x1^2*x2 + x1*x2^2
        sage: B.<t,x1,x2> = QQ['t,x1,x2']
        sage: demazure_pi_i(1, t*x1^2*x2)
        t*x1^2*x2 + t*x1*x2^2
    """
    br = parent(f)
    x = alphabet
    return divided_difference(i, br(x+str(i))*f, alphabet=x)

def demazure_pi_w(w, f, alphabet='x', offset=0):
    r"""
    Apply the (isobaric) Demazure `\pi_w` operator indexed by permutation `w` to a multivariate function ``f``.

    Note, `\pi_w = \pi_{i_1} \cdots \pi_{i_l}` where `w = s_{i_1} \cdots s_{i_l}`.

    EXAMPLES::

        sage: A.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: demazure_pi_w([1,2,3], x1^2*x2) # does nothing
        x1^2*x2
        sage: demazure_pi_w([2,1,3], x1^2*x2) # same as demazure_pi_i(1, x1^2*x2)
        x1^2*x2 + x1*x2^2
        sage: demazure_pi_w([3,1,2], x1^2*x2) == demazure_pi_i(2, demazure_pi_i(1, x1^2*x2))
        True
    """
    return _iterate_operators_from_reduced_word(demazure_pi_i, w, f, alphabet=alphabet, offset=offset)

def demazure_pi_i_on_polynomial(i, poly, alphabet='x'):
    r"""
    Apply the Demazure `\pi_i` operator to a multivariate polynomial.

    EXAMPLES::
        sage: A.<x1,x2> = QQ['x1,x2']
        sage: demazure_pi_i_on_polynomial(1, x1^2*x2)
        x1^2*x2 + x1*x2^2
        sage: B.<t,x1,x2> = QQ['t,x1,x2']
        sage: demazure_pi_i_on_polynomial(1, t*x1^2*x2)
        t*x1^2*x2 + t*x1*x2^2
    """
    br = parent(poly)
    x = alphabet
    return divided_difference_on_polynomial(i, br(x+str(i))*poly, alphabet=x)

def demazure_pi_w_on_polynomial(w, poly, alphabet='x'):
    return _iterate_operators_from_reduced_word(demazure_pi_i_on_polynomial, w, poly, alphabet=alphabet)
   
def key_polynomial(alph, alphabet='x', ambient_ring=None):
    r"""
    Return the key polynomial or Demazure character indexed by integer vector ``alph``.

    Note, SageMath has a built-in library for key polynomials, but it does not provide the more general Laurent polynomials.

    See `sage.combinat.key_polynomial.KeyPolynomial`.

    EXAMPLES::

        sage: key_polynomial([2,1,0])
        x1^2*x2
        sage: key_polynomial([1,2])
        x1^2*x2 + x1*x2^2
        sage: key_polynomial([-2,-1])
        x1^-1*x2^-2 + x1^-2*x2^-1
    """
    if not ambient_ring:
        ambient_ring = generate_laurent_polynomial_ring(Rationals(), len(alph), alphabet)
    w = Word([-a for a in alph]).standard_permutation().inverse()
    sorted_alph = list(reversed(sorted(alph)))
    monomial = prod([ambient_ring(alphabet+str(i+1))**(sorted_alph[i]) for i in range(len(alph))])
    return demazure_pi_w_on_polynomial(w, monomial, alphabet)

def demazure_pi_i_hat(i, poly, alphabet='x'):
    return demazure_pi_i(i, poly, alphabet)-poly

def demazure_pi_w_hat(w, poly, alphabet='x'):
    return _iterate_operators_from_reduced_word(demazure_pi_i_hat, w, poly, alphabet=alphabet)

def demazure_pi_i_hat_on_polynomial(i, poly, alphabet='x'):
    return demazure_pi_i_on_polynomial(i, poly, alphabet)-poly

def demazure_pi_w_hat_on_polynomial(w, poly, alphabet='x'):
    return _iterate_operators_from_reduced_word(demazure_pi_i_hat_on_polynomial, w, poly, alphabet=alphabet)

def atom_polynomial(alph, alphabet='x', ambient_ring=None):
    r"""
    Return the atom polynomial indexed by integer vector ``alph``.

    EXAMPLES::

        sage: atom_polynomial([3,2,1])
        x1^3*x2^2*x3
        sage: atom_polynomial([-3,0,0])
        x1^-1*x3^-2 + x1^-1*x2^-1*x3^-1 + x1^-2*x3^-1 + x1^-1*x2^-2 + x1^-2*x2^-1 + x1^-3
    """
    if not ambient_ring:
        ambient_ring = generate_laurent_polynomial_ring(Rationals(), len(alph), alphabet)
    w = Word([-a for a in alph]).standard_permutation().inverse()
    sorted_alph = list(reversed(sorted(alph)))
    monomial = prod([ambient_ring(alphabet+str(i+1))**(sorted_alph[i]) for i in range(len(alph))])
    return ambient_ring(demazure_pi_w_hat(w, monomial, alphabet))

def _setup_demazure_inner_prod_milp_on_monomial_exponents(alpha, beta):
    P = MixedIntegerLinearProgram()
    x = P.new_variable(integer=True, nonnegative=True)
    n = len(beta)-1
    RS = RootSystem(['A',n])
    ZZnp1 = RS.ambient_space()
    pos_roots = [ZZnp1(r) for r in RS.root_lattice().positive_roots()]
    for k in range(len(beta)):
        P.add_constraint(sum(x[i]*pos_roots[i][k] for i in range(len(pos_roots))) + beta[k]+alpha[k] == 0)
    for i in range(len(pos_roots)):
        P.add_constraint(x[i] <= 1)
    return P

def _demazure_inner_prod_on_monomial_exponents(alpha, beta):
    if len(alpha) != len(beta):
        alpha = list(alpha) + [0]*(len(beta)-len(alpha))
        beta = list(beta) + [0]*(len(alpha)-len(beta))
    milp = _setup_demazure_inner_prod_milp_on_monomial_exponents(alpha,beta) 
    return sum((-1)**sum(soln) for soln in milp.polyhedron(base_ring=QQ, backend='normaliz').integral_points())

def demazure_inner_prod(f, g, alphabet='x'):
    r"""
    Return the result of the Demazure inner product `\langle f, g \rangle_0`.

    Note, this code takes as its definition `\langle f(z), g(z) \rangle_0 = \langle z^0 \rangle f(z) \cdot g (z) \prod_{1 \leq i < j \leq l} (1-z_i/z_j)`.

    EXAMPLES::

        sage: key_poly = key_polynomial([2,0,1])
        sage: atom_poly = atom_polynomial([-2,0,-1])
        sage: demazure_inner_prod(key_poly, atom_poly)
        1
        sage: atom_poly2 = atom_polynomial([0,-2,-1])
        sage: demazure_inner_prod(key_poly, atom_poly2)
        0
        sage: A = generate_laurent_polynomial_ring(QQ, 3, pre_extra_vars=['q','t'])
        sage: t = A('t')
        sage: q = A('q')
        sage: demazure_inner_prod(t*key_polynomial([2,0,1],ambient_ring=A),q^2*atom_polynomial([-2,0,-1],ambient_ring=A))
        q^2*t
    """
    Af = parent(f)
    fgens = Af.gens()
    fxx = [x for x in fgens if str(x)[0] == alphabet]
    Ag = parent(g)
    ggens = Ag.gens()
    gxx = [x for x in ggens if str(x)[0] == alphabet]
    f_coeff_mons = separate_polynomial_generators(fxx, f)
    g_coeff_mons = separate_polynomial_generators(gxx, g)
    return sum(coeff1*coeff2*_demazure_inner_prod_on_monomial_exponents(mon1.exponents()[0], mon2.exponents()[0]) for ((coeff1,mon1),(coeff2,mon2)) in cartesian_product([f_coeff_mons,g_coeff_mons]))

def demazure_lusztig_i(i, f, alphabet='x', v=None, convention='hhl'):
    r"""
    Give the action of the Demazure-Lusztig operator `T_i` on `f` by definition.

    Note, Sage has multiple implementations of Demazure-Lusztig operators already.

    The convention ``'hhl'`` corresponds to the Haglund-Haiman-Loehr conventions used
    in "Flagged LLT polynomials, nonsymmetric plethysm, and nonsymmetric Macdonald polynomials."

    The convention ``'paths'`` corresponds to the conventions used in
    "A Shuffle Tehorem for Paths Under Any Line" except in `t` instead of `q`.

    EXAMPLES::

        sage: R.<t,x1,x2,x3> = QQ['t,x1,x2,x3']
        sage: demazure_lusztig_i(1,t*x1^2*x2) 
        t*x1*x2^2
        sage: demazure_lusztig_i(1,x1*x2^2)
        t*x1^2*x2 + t*x1*x2^2 - x1*x2^2
        sage: from polynomial_utils import monomial_basis_in_fixed_xy_degree
        sage: mons = monomial_basis_in_fixed_xy_degree(3,0,R,'x','t')
        sage: all([demazure_lusztig_i(1,demazure_lusztig_i(1,mon)+mon)-t*(demazure_lusztig_i(1,mon)+mon) == 0 for mon in mons])
        True
        sage: all([demazure_lusztig_i(2,demazure_lusztig_i(2,mon)+mon)-t*(demazure_lusztig_i(2,mon)+mon) == 0 for mon in mons])
        True
        sage: demazure_lusztig_i(1,x1, convention='paths')
        t*x1 + t*x2 - x1
        sage: all([demazure_lusztig_i(1,demazure_lusztig_i(1,mon,convention='paths')+mon,convention='paths')-t*(demazure_lusztig_i(1,mon,convention='paths')+mon) == 0 for mon in mons])
        True
        sage: all([demazure_lusztig_i(2,demazure_lusztig_i(2,mon,convention='paths')+mon,convention='paths')-t*(demazure_lusztig_i(2,mon,convention='paths')+mon) == 0 for mon in mons])
        True
    """
    br = parent(f)
    if not v:
        v = br('t')
    if convention == 'hhl':
        return (1-v)*(demazure_pi_i(i, f, alphabet=alphabet)-f)+v*(s_i(br, i, alphabet=alphabet)(f))
    elif convention == 'paths':
        return (v-1)*demazure_pi_i(i, f, alphabet=alphabet)+s_i(br, i, alphabet=alphabet)(f)

def demazure_lusztig_i_inverse(i, f, alphabet='x', v=None, convention='hhl'):
    r"""
    Return the result of the inverse Demazure-Lusztig operator of ``f``

    EXAMPLES::

        sage: R.<t,x1,x2,x3> = QQ['t,x1,x2,x3']
        sage: from polynomial_utils import monomial_basis_in_fixed_xy_degree
        sage: mons = monomial_basis_in_fixed_xy_degree(3,0,R,'x','t')
        sage: all([demazure_lusztig_i(1,demazure_lusztig_i_inverse(1,mon)) == mon for mon in mons])
        True
        sage: all([demazure_lusztig_i_inverse(1,demazure_lusztig_i(1,mon)) == mon for mon in mons])
        True
        sage: all([demazure_lusztig_i(1,demazure_lusztig_i_inverse(1,mon,convention='paths'),convention='paths') == mon for mon in mons])
        True
        sage: all([demazure_lusztig_i_inverse(1,demazure_lusztig_i(1,mon,convention='paths'),convention='paths') == mon for mon in mons])
        True
    """
    if not v:
        par = parent(f)
        v = par('t')
    return v**(-1)*(demazure_lusztig_i(i, f, alphabet=alphabet, v=v, convention=convention) + (1-v)*f)

def demazure_lusztig_w(w, f, offset=0, alphabet='x', v=None, convention='hhl'):
    return _iterate_operators_from_reduced_word(demazure_lusztig_i, w, f, offset=offset, alphabet=alphabet, v=v, convention=convention)

def demazure_lusztig_i_on_polynomial(i, poly,  alphabet='x', v=None, convention='hhl'):
    br = parent(poly)
    if not v:
        v = br('t')
    if convention == 'hhl':
        return (1-v)*(demazure_pi_i_on_polynomial(i, poly, alphabet=alphabet)-poly)+v*(s_i_on_polynomial(i, poly, alphabet=alphabet))
    elif convention == 'paths':
        return (v-1)*demazure_pi_i_on_polynomial(i, poly, alphabet=alphabet)+s_i_on_polynomial(i, poly, alphabet=alphabet)

def demazure_lusztig_i_inverse_on_polynomial(i, poly, alphabet='x', v=None, convention='hhl'):
    if not v:
        par = parent(poly)
        v = par('t')
    return v**(-1)*(demazure_lusztig_i(i, poly, alphabet=alphabet, v=v, convention=convention) + (1-v)*poly)

def demazure_lusztig_w_on_polynomial(w, poly, alphabet='x', v=None, convention='hhl', offset=0):
    r"""
    Return the action of Demazure-Lusztig operator `T_i` on polynomial ``poly``.

    EXAMPLES::

        sage: R.<t,x1,x2,x3> = QQ['t,x1,x2,x3']
        sage: demazure_lusztig_w_on_polynomial([2,1,3], t*x1^2*x2)
        t*x1*x2^2
    """
    return _iterate_operators_from_reduced_word(demazure_lusztig_i_on_polynomial, w, poly, offset=offset, alphabet=alphabet, v=v, convention=convention)

def hecke_symmetrize_over_perms(perms, poly, alphabet='x', v=None, convention='hhl', perm_coeff_dict = None):
    r"""
    Given a list of permutations ``perms``, denoted `S`, return the result of `\sum_{w \in S} T_w` applied to ``poly``.
    """
    perms = [Permutation(perm) for perm in perms]
    br = parent(poly)
    if not perm_coeff_dict:
        perm_coeff_dict = {perm:br.one() for perm in perms}
    return sum(perm_coeff_dict[perm]*demazure_lusztig_w_on_polynomial(perm, poly, alphabet=alphabet, v=v, convention=convention) for perm in perms)

def nonsymmetric_hall_littlewood_E(alph, twist=None, v=None, alphabet='x', ambient_ring=None):
    r"""
    Returns the nonsymmetric Hall-Littlewood E polynomial indexed by ``alph`` and twisted by ``twist``.

    EXAMPLES::

        sage: nonsymmetric_hall_littlewood_E([0,0,0])
        1
        sage: nonsymmetric_hall_littlewood_E([1,0,0])
        x1
        sage: nonsymmetric_hall_littlewood_E([0,1,0])
        x1 + x2 - t^-1*x1
        sage: nonsymmetric_hall_littlewood_E([0,0,1])
        x1 + x2 + x3 - t^-1*x1 - t^-1*x2
        sage: nonsymmetric_hall_littlewood_E([1,1,0])
        x1*x2
        sage: nonsymmetric_hall_littlewood_E([1,0,1])
        x1*x2 + x1*x3 - t^-1*x1*x2
        sage: nonsymmetric_hall_littlewood_E([0,1,1])
        x1*x2 + x1*x3 + x2*x3 - t^-1*x1*x2 - t^-1*x1*x3
        sage: nonsymmetric_hall_littlewood_E([2,0,0])
        x1^2
        sage: nonsymmetric_hall_littlewood_E([0,2,0])
        x1^2 + x1*x2 + x2^2 - t^-1*x1^2 - t^-1*x1*x2
        sage: A.<q> = QQ['q']
        sage: nonsymmetric_hall_littlewood_E([0,1,1], v=q)
        x1*x2 + x1*x3 + x2*x3 - q^-1*x1*x2 - q^-1*x1*x3
        sage: nonsymmetric_hall_littlewood_E([0,1,1], v=q**(-1))
        -q*x1*x2 - q*x1*x3 + x1*x2 + x1*x3 + x2*x3 
        sage: nonsymmetric_hall_littlewood_E([1,0,0], twist=[3,2,1])
        x1
        sage: nonsymmetric_hall_littlewood_E([0,2,0],alphabet='y')
        y1^2 + y1*y2 + y2^2 - t^-1*y1^2 - t^-1*y1*y2
        sage: from polynomial_utils import generate_multi_laurent_polynomial_ring
        sage: A = generate_multi_laurent_polynomial_ring(QQ, 3, pre_extra_vars=['t'])
        sage: nonsymmetric_hall_littlewood_E([1,0,1],alphabet='y',ambient_ring=A)*nonsymmetric_hall_littlewood_E([0,1,0],alphabet='x',ambient_ring=A)
        x1*y1*y2 + x2*y1*y2 + x1*y1*y3 + x2*y1*y3 - 2*t^-1*x1*y1*y2 - t^-1*x2*y1*y2 - t^-1*x1*y1*y3 + t^-2*x1*y1*y2

    The nonsymmetric Hall-Littlewood `E(x;t^{-1})` polynomials specialize to key polynomials (or Demazure characters) at `t = 0`.

    ::

        sage: A = Frac(QQ['t'])
        sage: f = nonsymmetric_hall_littlewood_E([1,0,1],v=A('t')**(-1))
        sage: B = parent(f)
        sage: t = B('t')
        sage: f.subs({t:0}) 
        x1*x2 + x1*x3
        sage: all([specialize_flat_polynomial_variables({t:0},nonsymmetric_hall_littlewood_E(list(alph),v=t**(-1))) == key_polynomial(list(alph), ambient_ring=B) for alph in IntegerVectors(2,3)])
        True

    Note, the nonsymmetric Hall-Littlewood `E(x;t^{-1})` polynomials are `q=0` specializations of nonsymmetric Macdonald E polynomials as implemented in Sage.

    ::

        sage: from sage.combinat.sf.ns_macdonald import E
        sage: K = Frac(QQ['q,t'])
        sage: A = generate_laurent_polynomial_ring(K,3)
        sage: A(sum(coeff.subs({'q':0})*mon for (coeff,mon) in E([0,1,1])))
        (-t + 1)*x1*x2 + (-t + 1)*x1*x3 + x2*x3
        sage: nonsymmetric_hall_littlewood_E([0,1,1],v=A('t')**(-1),ambient_ring=A)
        (-t + 1)*x1*x2 + (-t + 1)*x1*x3 + x2*x3
        sage: all(A(sum(coeff.subs({'q':0})*mon for (coeff,mon) in E(list(alph)))) == nonsymmetric_hall_littlewood_E(list(alph),v=A('t')**(-1),ambient_ring=A) for alph in IntegerVectors(2,3))
        True
    """
    if not ambient_ring:
        if not v:
            ambient_ring = generate_laurent_polynomial_ring(Rationals(), len(alph), alphabet, pre_extra_vars = ['t'])
        else:
            coeff_gens = [str(q) for q in parent(v).gens() if str(q)[0] != alphabet] 
            ambient_ring = generate_laurent_polynomial_ring(Rationals(), len(alph), alphabet, pre_extra_vars = list(coeff_gens))
            v = ambient_ring(v)
    if not v:
        v = ambient_ring('t')
    if not twist:
        twist = Permutation(list(range(1,len(alph)+1)))
    w = Word([-a for a in alph]).standard_permutation().inverse()
    w_red_word = w.reduced_word()
    sorted_alph = list(reversed(sorted(alph)))
    monomial = prod([ambient_ring(alphabet+str(i+1))**(sorted_alph[i]) for i in range(len(alph))])
    perms = Permutations(len(alph))
    sigma_red_word = perms(twist).reduced_word()
    if not w_red_word:
        return monomial
    elif not sigma_red_word:
        return ambient_ring(v)**(-len(w_red_word))*demazure_lusztig_w_on_polynomial(w, monomial, alphabet=alphabet, v=v, convention='paths')
    else:
        i = sigma_red_word[0]
        si_alph = alph[:i-1] + [alph[i],alph[i-1]] + alph[i+1:]
        si_sigma = perms.from_reduced_word(sigma_red_word[1:]) 
        return v**(alph[i-1] >= alph[i])*demazure_lusztig_i_inverse_on_polynomial(i, nonsymmetric_hall_littlewood_E(si_alph, twist=si_sigma, v=v, alphabet=alphabet, ambient_ring=ambient_ring), alphabet=alphabet, v=v, convention='paths')

def nonsymmetric_hall_littlewood_F(alph, twist=None, v=None, alphabet='x', ambient_ring=None):
    r"""
    Returns the nonsymmetric Hall-Littlewood F polynomial indexed by ``alph`` and twisted by ``twist``.

    EXAMPLES::

        sage: nonsymmetric_hall_littlewood_F([0,0,0])
        1
        sage: nonsymmetric_hall_littlewood_F([1,0,0])
        x1
        sage: nonsymmetric_hall_littlewood_F([0,1,0])
        x2
        sage: nonsymmetric_hall_littlewood_F([0,0,1])
        x3
        sage: nonsymmetric_hall_littlewood_F([1,1,0])
        x1*x2
        sage: nonsymmetric_hall_littlewood_F([1,0,1])
        x1*x3
        sage: nonsymmetric_hall_littlewood_F([0,1,1])
        x2*x3
        sage: nonsymmetric_hall_littlewood_F([2,0,0])
        -t*x1*x2 - t*x1*x3 + x1^2 + x1*x2 + x1*x3
        sage: nonsymmetric_hall_littlewood_F([0,2,0])
        -t*x2*x3 + x2^2 + x2*x3
        sage: nonsymmetric_hall_littlewood_F([0,0,2])
        x3^2
        sage: nonsymmetric_hall_littlewood_F([2,0,0]) == invert_variables_in_flat_polynomial(nonsymmetric_hall_littlewood_E([-2,0,0],[3,2,1]))
        True
        sage: nonsymmetric_hall_littlewood_F([0,2,0],[2,3,1])
        -t*x2*x3 + x2^2 + x2*x3

    The nonsymmetric Hall-Littlewood F polynomials specialize at `t=0` to opposite Demazure atoms. 

    ::

        sage: f = nonsymmetric_hall_littlewood_F([0,2,0])
        sage: A = parent(f)
        sage: t = A('t')
        sage: f.subs({t:0})
        x2^2 + x2*x3
        sage: invert_variables_in_flat_polynomial(atom_polynomial([0,-2,0]))
        x2^2 + x2*x3
        sage: all(nonsymmetric_hall_littlewood_F(list(alph)).subs({t:0}) == invert_variables_in_flat_polynomial(atom_polynomial([-a for a in alph],ambient_ring=A)) for alph in IntegerVectors(2,3))
        True
    """
    if not ambient_ring:
        stv = str(v) if v else 't'
        ambient_ring = generate_laurent_polynomial_ring(Rationals(), len(alph), alphabet, pre_extra_vars = [stv])
    if not v:
        v = ambient_ring('t')
    if not twist:
        twist = Permutation(list(range(1,len(alph)+1)))
    w = Word(alph).standard_permutation().inverse()
    sorted_alph = list(sorted(alph))
    monomial = prod([ambient_ring.gens_dict()[alphabet+str(i+1)]**(sorted_alph[i]) for i in range(len(alph))])
    perms = Permutations(len(alph))
    sigma_red_word = perms(twist).reduced_word()
    if not sigma_red_word:
        return demazure_lusztig_w_on_polynomial(w, monomial, alphabet=alphabet, v=v, convention='paths')
    else:
        l = len(alph)
        neg_alph = [-a for a in alph]
        w0twist_sigma = Permutation([twist[l-1-i] for i in range(l)])
        prebar = nonsymmetric_hall_littlewood_E(neg_alph, twist=w0twist_sigma, v=v**(-1), alphabet=alphabet, ambient_ring=ambient_ring)
        prebar_xx = [x for x in parent(prebar).gens() if str(x)[0] == alphabet]
        return invert_variables_in_flat_polynomial(prebar, var_list=prebar_xx) 

