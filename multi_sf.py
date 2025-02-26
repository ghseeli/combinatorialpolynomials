# ****************************************************************************
#  Copyright (C) 2025 George H. Seelinger <ghseeli@gmail.com>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#                  https://www.gnu.org/licenses/
# ***************************************************************************

r"""
Utilities for dealing with symmetric functions in multiple sets of alphabets.

At the primitive level, the data for these functions are lists of tuples of the form (tuple of symmetric functions, coefficient).
However, this representation is not unique, and so the preferred method to deal with these functions is to use the `SymFnInAlphabetsRing`. 
"""

from sage.all import Algebras, CombinatorialFreeModule, Parent, parent, Partition, prod, Sets, SymmetricFunctions
from sage.structure.element import Element
from sage.misc.fast_methods import Singleton

def simplify_linear_comb_vs(list_of_tuples, br=None):
    r"""
    Given a list of tuples of the form (support, coefficient) representing a linear combination, simplify the list so that each support only appears once.

    EXAMPLES::

        sage: simplify_linear_comb_vs([((2,1),5),((0,3),2),((2,1),-2)])
        [((2, 1), 3), ((0, 3), 2)]
    """
    if not br:
        br = parent(list_of_tuples[0][1])
    basis_keys = [key for (key, coeff) in list_of_tuples]
    F = CombinatorialFreeModule(br, basis_keys)
    return list(sum([coeff*F(key) for (key, coeff) in list_of_tuples]))

def apply_sym_fn_map_to_ith_coord(multisym_f, phi, i):
    r"""
    Given the "list form" of a multisymmetric function, apply a symmetric function map ``phi`` to the ``i``th coordinate.

    EXAMPLES::

        sage: s = SymmetricFunctions(QQ).s()
        sage: multisym_f = [((s[1],s[2],s[1]),3)]
        sage: apply_sym_fn_map_to_ith_coord(multisym_f, lambda f: f.omega(), 1)
        [((s[1], s[1, 1], s[1]), 3)]
        sage: apply_sym_fn_map_to_ith_coord(multisym_f, lambda f: f*s[1], 2)
        [((s[1], s[2], s[1, 1]), 3), ((s[1], s[2], s[2]), 3)]
    """
    res = []
    for (tup, coeff) in multisym_f:
        ith_piece = tup[i]
        phi_ith_piece = phi(ith_piece)
        par = parent(phi_ith_piece)
        res = res + [(tuple(list(tup)[:i]+[par(supp)]+list(tup)[i+1:]),coeff*coeff2) for (supp,coeff2) in list(phi_ith_piece)]
    return simplify_linear_comb_vs(res)

def multisym_fn_mult(f1, f2):
    r"""
    Multiply two list form multisymmetric functions.

    EXAMPLES::

        sage: s = SymmetricFunctions(QQ).s()
        sage: multisym_f1 = [((s[2],s[1,1],s[2]),3),((s[1,1],s[2],s[2]),5)]
        sage: multisym_f2 = [((s[1],s[1],s[0]),2)]
        sage: multisym_fn_mult(multisym_f1, multisym_f2)
        [((s[2, 1], s[2, 1], s[2]), 16),
         ((s[2, 1], s[1, 1, 1], s[2]), 6),
         ((s[3], s[2, 1], s[2]), 6),
         ((s[3], s[1, 1, 1], s[2]), 6),
         ((s[2, 1], s[3], s[2]), 10),
         ((s[1, 1, 1], s[2, 1], s[2]), 10),
         ((s[1, 1, 1], s[3], s[2]), 10)]


        ..  SEEALSO::

            ``SymFnInAlphabetsRing``
    """
    res = []
    for (supp,coeff) in f1:
        for (supp2,coeff2) in f2:
            res = res + [(tuple([supp[i]*supp2[i] for i in range(len(supp))]),coeff*coeff2)]
    idmap = lambda f: f
    for i in range(len(f1[0][0])):
        res = apply_sym_fn_map_to_ith_coord(res, idmap, i)
    return res

class PartitionSequenceSpace(Singleton, Parent):
    r"""
    Utility function to detect whether a tuple of integer compositions is a tuple of partitions.
    """
    def __init__(self):
        Parent.__init__(self, facade=(tuple,), category=Sets().Infinite().Facade())

    def __contains__(self, seq):
        return (isinstance(seq, tuple) and all(i in Partitions() for i in seq))

def support_to_multisym_f(supp, basis):
    r"""
    Sends a tuple of partitions into a "multisymmetric function" format of a list of tuples, each with a tuple of symmetric functions and a coefficient.

    EXAMPLES::

        sage: s = SymmetricFunctions(QQ).s()
        sage: support_to_multisym_f(([2],[1,1],[1]), s)
        [((s[2], s[1, 1], s[1]), 1)]
        
    """
    return [(tuple(basis(pa) for pa in supp),1)]

def multisym_f_to_support(multisym_f):
    r"""
    Sends a tuple representation of a "multisymmetric function" to just the supporting partitions, effectively forgetting the basis.

    EXAMPLES::

        sage: s = SymmetricFunctions(QQ).s()
        sage: multisym_f_to_support([((s[2], s[1,1], s[1]),3)])
        [(([2], [1, 1], [1]), 3)]
    """
    return [(tuple(supp for elm in elml for (supp,coeff) in list(elm)),coeff2) for (elml,coeff2) in multisym_f]

class SymFnInAlphabetsRing(CombinatorialFreeModule):
    r"""
    Helper function representing products of symmetric functions in a list of alphabets, that is, for alphabets X_1, X_2, \ldots, X_l, this attempts to implement the ring `\Lambda(X_1) \otimes \Lambda(X_2) \otimes \cdots \otimes \Lambda(X_l)` in a specific symmetric function basis.

    EXAMPLES::

        sage: K = Frac(QQ['q,t'])
        sage: A.<X1,X2,X3> = K['X1,X2,X3']
        sage: s = SymmetricFunctions(K).s()
        sage: multi_s = SymFnInAlphabetsRing(K,s,(X1,X2,X2+X3))
        sage: multi_s([[2],[1,1],[1]])
        s[2](X1)*s[1, 1](X2)*s[1](X2 + X3)
        sage: multi_s([[1],[0],[0]])*multi_s([[2],[1,1],[1]])
        s[2, 1](X1)*s[1, 1](X2)*s[1](X2 + X3) + s[3](X1)*s[1, 1](X2)*s[1](X2 + X3)
        sage: multi_s.one()
        s[](X1)*s[](X2)*s[](X2 + X3)
    """
    def __init__(self, base_ring, sf_basis, alphabets):
        category = Algebras(base_ring.category()).WithBasis()
        category = category.or_subcategory(category)
        indices = PartitionSequenceSpace()
        CombinatorialFreeModule.__init__(self, base_ring, indices, category=category)
        self.sf_basis = sf_basis
        self.alphabets = tuple(alphabets)

    def _repr_(self):
        return repr(self.sf_basis) + " in alphabets " +repr(self.alphabets)

    def _prepare_seq_(self, seq):
        return tuple(Partition(s) for s in seq)
    
    def _element_constructor_(self, seq):
        if seq in self.base_ring():
            return self.term(self.one_basis(), self.base_ring()(seq))
        return self.monomial(self._prepare_seq_(seq)) 

    def is_field(self):
        return False

    def one_basis(self):
        return tuple([Partition([])]*len(self.alphabets))
    
    def product_on_basis(self, left, right):
        basis = self.sf_basis
        l1 = len(left)
        l2 = len(right)
        left_in_basis = [(tuple([basis(pa) for pa in left]+[basis.one() for i in range(l2-l1)]),1)]
        right_in_basis = [(tuple([basis(pa) for pa in right] + [basis.one() for i in range(l1-l2)]),1)]
        res = multisym_fn_mult(left_in_basis, right_in_basis)
        res_tups = [(tuple(supp for elm in elml for (supp,coeff) in list(elm)),coeff2) for (elml,coeff2) in res]
        return sum([coeff*self.monomial(supp) for (supp,coeff) in res_tups])

    def _repr_term(self, term):
        pref = self.sf_basis.prefix()
        alphabets = self.alphabets
        strs = [pref+repr(term[i])+"("+repr(alphabets[i])+")" for i in range(len(term))]
        return reduce(lambda a,b: a + "*" + b, strs) 

    def formal_plethysm(self, module_map):
        r"""
        Simply replaces the alphabets A_i with new alphabets f(A_i) when given module map f: A \to B.

        In otherwords, `g_1(A_1) \otimes \cdots \otimes g_r(A_r) \mapsto g_1(f(A_1)) \otimes \cdots \otimes g_r(f(A_r))`.
        """
        codom = SymFnInAlphabetsRing(self.base_ring(),self.sf_basis,tuple(module_map(alph) for alph in self.alphabets))
        return self.module_morphism(lambda z: codom(z), codomain=codom)

    def map_basis_to_poly_in_alphabets(self, supp):
        r"""
        Given a tuple of partitions, return the polynomial evaluation of the multisymmetric function in those alphabets.

        Note, this assumes each summand in the alphabet is a single variable, which may or may not be your intention for what the variable represents.

        EXAMPLES::

            sage: K = Frac(QQ['q,t'])
            sage: A.<x1,x2,x3> = K['x1,x2,x3']
            sage: s = SymmetricFunctions(K).s()
            sage: multi_s = SymFnInAlphabetsRing(K,s,(x1,x2,x2+x3))
            sage: multi_s.map_basis_to_poly_in_alphabets([[2],[1],[1,1]])
            x1^2*x2^2*x3
            sage: multi_s.map_basis_to_poly_in_alphabets([[2,1],[1],[1]])
            0

        Notice, the last call returns zero because `s_{21}(x1) = 0`.

        We test that the code works also when the base ring is not a polynomial ring ::

            sage: K = QQ
            sage: A.<x1,x2,x3> = K['x1,x2,x3']
            sage: s = SymmetricFunctions(K).s()
            sage: multi_s = SymFnInAlphabetsRing(K,s,(x1,x2,x2+x3))
            sage: multi_s.map_basis_to_poly_in_alphabets([[2],[1],[1,1]])
            x1^2*x2^2*x3
        """
        from sage.rings.polynomial.flatten import FlatteningMorphism
        A = parent(self.alphabets[0])
        Agens = A.gens()
        base_gens = self.base_ring().gens()
        if base_gens != (1,):
            A_flat = Frac(PolynomialRing(A.base_ring(),len(Agens)+len(base_gens),list(Agens)+list(base_gens)))
        else:
            A_flat = A
        codom = getattr(SymmetricFunctions(A_flat),self.sf_basis.prefix())()
        factors = [codom(supp[i]).plethysm(A_flat(str(self.alphabets[i]))*codom.one()) for i in range(len(supp))]
        return A((prod(factors)*codom.one()).coefficient([]))
        
    def map_to_poly_in_alphabets(self):
        r"""
        A map from ``self`` to the alphabet ring given by plethystically evaluating each symmetric function, assuming monomials in the alphabet are actual monomials.

        EXAMPLES::

            sage: K = Frac(QQ['q,t'])
            sage: A.<x1,x2,x3> = K['x1,x2,x3']
            sage: s = SymmetricFunctions(K).s()
            sage: multi_s = SymFnInAlphabetsRing(K,s,(x1,x2,x2+x3))
            sage: poly_eval_map = multi_s.map_to_poly_in_alphabets()
            sage: poly_eval_map(multi_s([[2],[1],[3]])+multi_s([[3],[3],[0]]))
            (x1^3*x2^3+x1^2*x2^4+x1^2*x2^3*x3+x1^2*x2^2*x3^2+x1^2*x2*x3^3)*s[]

        ..  SEEALSO::

            ``pleth_eval_multi_sym_fn``
        """
        return self.module_morphism(self.map_basis_to_poly_in_alphabets, codomain=getattr(SymmetricFunctions(parent(self.alphabets[0])),self.sf_basis.prefix())())


def pleth_eval_multi_sym_fn(fn, eval_list):
    r"""
    Given a multi symmetric function ``fn`` in polynomials of alphabets ``A_i``, evaluate the polynomial given by specifying each ``A_i`` to some plethystic quantity. 

    EXAMPLES::

        sage: K = Frac(QQ['q,t'])
        sage: A.<X1,X2,X3> = K['X1,X2,X3']
        sage: s = SymmetricFunctions(K).s()
        sage: multi_s = SymFnInAlphabetsRing(K,s,(X1,X2,X3))
        sage: B.<x1,x2,x3,x4,x5,x6> = K['x1,x2,x3,x4,x5,x6']
        sage: pleth_eval_multi_sym_fn(multi_s([[2],[1,1],[1]]),[x1+x2,x3+x4,x5+x6]) == (x1^2+x1*x2+x2^2)*(x3*x4)*(x5+x6)
        True

    """
    par = parent(fn)
    alphabet_R = parent(par.alphabets[0])
    pleth_res = par.formal_plethysm(alphabet_R.hom(eval_list))(fn)
    par_res = parent(pleth_res)
    return (par_res.map_to_poly_in_alphabets()(pleth_res)).coefficient([])
