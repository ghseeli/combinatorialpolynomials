from sage.all import matrix, MatrixSpace, parent, QQ, vector, block_matrix
import bisect
from itertools import chain
from functools import reduce
import warnings

class LazyBlockMatrix:
    r"""
    Data structure for a block matrix, but it is lazy in the sense that it does not actually store the array structure. 

    EXAMPLES::

        sage: A = matrix(2,4,{(0,1):2,(1,3):3})
        sage: B = matrix(3,5,{(0,0):5, (2,4):7})
        sage: C = matrix(2,5,{})
        sage: D = matrix(3,4,{})
        sage: T = LazyBlockMatrix([[A,C],[D,B]])
        sage: T.entry(0,1)
        2
        sage: T.entry(4,8)
        7
        sage: T.entry(0,7)
        0
        sage: v = vector([1,2,3,4,5,6,7,8,9])
        sage: T.apply_to_vector(v) == matrix(5,9,{(0,1):2, (1,3):3, (2,4):5, (4,8):7})*v
        True
        
    """
    def __init__(self, mat_blocks):
        self.blocks = mat_blocks
        self.block_nrows = [row[0].nrows() for row in self.blocks]
        self.block_ncols = [mat.ncols() for mat in self.blocks[0]]
        self._block_nrow_p_sum = [sum(self.block_nrows[:i]) for i in range(1,len(self.block_nrows)+1)]
        self._block_ncol_p_sum = [sum(self.block_ncols[:i]) for i in range(1,len(self.block_ncols)+1)]

    def __repr__(self):
        return "LazyBlockMatrix of " + str(self.blocks)

    def check(self):
        block_nrows = [[mat.nrows() for mat in row] for row in self.blocks]
        row_conds = all([len(list(set(nrow))) == 1 for nrow in block_nrows])
        block_ncols = [[mat.ncols() for mat in row] for row in self.blocks]

    def nrows(self):
        return sum(self.block_nrows)

    def ncols(self):
        return sum(self.block_ncols)

    def dict(self):
        return {(i,j):self.blocks[i][j] for i in range(len(self.blocks)) for j in range(len(self.blocks[i]))}

    def entry(self, i,j):
        block_row_ind = bisect.bisect_right(self._block_nrow_p_sum, i)
        block_col_ind = bisect.bisect_right(self._block_ncol_p_sum, j)
        return self.blocks[block_row_ind][block_col_ind][i-self._block_nrow_p_sum[block_row_ind]][j-self._block_ncol_p_sum[block_col_ind]]

    def __getitem__(self, i, j):
        return self.entry(i,j)

    def __mul__(self, other):
        assert other.ncols() == self.nrows(), "Cannot multiply " " matrix * matrix with " " columns!"
        
            
    def apply_to_vector(self, vec):
        vector_blocks = [vector(vec[0:self._block_ncol_p_sum[0]])]+[vector(vec[self._block_ncol_p_sum[i]:self._block_ncol_p_sum[i+1]]) for i in range(len(self._block_ncol_p_sum)-1)]
        res_block_vec = [sum([self.blocks[i][j]*vector_blocks[j] for j in range(len(self.blocks[i]))]) for i in range(len(self.blocks))]
        return vector(reduce(lambda a,b: chain(a,b), res_block_vec))

    def to_matrix(self, subdivide=False, sparse=False):
        return block_matrix(self.blocks, subdivide=subdivide, sparse=sparse)


def tensor_product_of_sparse_rational_matrices(A,B):
    r"""
    Return the tensor product of two sparse rational matrices.

    EXAMPLES::

        sage: M1 = MatrixSpace(QQ,2,3)
        sage: A = M1({(0,0):3,(1,2):7})
        sage: M2 = MatrixSpace(QQ,3,4)
        sage: B = M2({(0,2):2, (2,3):5})
        sage: tensor_product_of_sparse_rational_matrices(A,B) == A.tensor_product(B)
        True
        sage: parent(tensor_product_of_sparse_rational_matrices(A,B))
        Full MatrixSpace of 6 by 12 sparse matrices over Rational Field
    """
    Adict = A.dict()
    Bdict = B.dict()
    Brows = B.nrows()
    Bcols = B.ncols()
    nonzero_blocks = [((i,j),Adict[(i,j)]*B) for (i,j) in Adict.keys()]
    flatten_blocks = {(i*Brows+ip,j*Bcols+jp):block[(ip,jp)] for ((i,j),block) in nonzero_blocks for (ip,jp) in Bdict.keys()}
    M = MatrixSpace(QQ,A.nrows()*B.nrows(),A.ncols()*B.ncols(), sparse=True)
    return M(flatten_blocks)

def element_to_vector(encoding, elm_tup_list, base_ring=None, sparse=None):
    r"""
    Given a list of tuples of the form (coefficient, basis_vector) and an encoding from basis vectors to indices, return the vector representing the ``elm_tup_list``.

    This is meant to be a very abstract and flexible function, capable of handling many different kinds of inputs.

    EXAMPLES::

        sage: A.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: id_op = lambda z : z
        sage: encoding = {x1:0, x2:1, x3:2}
        sage: element_to_vector(encoding, [(2,x1), (-3, x2), (1, x3)])
        (2, -3, 1)
        sage: encoding2 = {(1,0,0):0, (0,1,0):1, (0,0,1):2}
        sage: element_to_vector(encoding2, [(-3,(0,1,0)),(1,(0,0,1)),(2,(1,0,0))])
        (2, -3, 1)
        sage: from sage.all import Partitions, SymmetricFunctions
        sage: sym = SymmetricFunctions(QQ)
        sage: s = sym.s()
        sage: parts = Partitions(3)
        sage: encoding = {s(parts[i]):i for i in range(len(parts))}
        sage: element_to_vector(encoding, [(-3, s[2,1]),(2,s[3]),(1, s[1,1,1])])
        (2, -3, 1)
    """
    if not base_ring:
        base_ring = parent(elm_tup_list[0][0])
    if not sparse:
        list_vec = [0]*len(encoding)
        for (coeff, bi) in elm_tup_list:
            list_vec[encoding[bi]] = coeff
        return vector(base_ring, list_vec)
    else:
        max_encode = len(encoding)
        pre_vec = {encoding[bi]:coeff for (coeff,bi) in elm_tup_list}
        if max_encode-1 not in pre_vec.keys():
            pre_vec[max_encode-1] = 0
        return vector(base_ring, pre_vec, sparse=sparse)

def vector_to_element(decoding, vec, sparse=True):
    r"""
    Convert a vector back to an element represented as a list of (coefficient, basis_element) tuples.

    This is the inverse operation to ``element_to_vector``. Given a vector of coefficients
    and a decoding that maps indices back to basis elements, reconstruct the element
    in list form.

    EXAMPLES:: 

        sage: from sage.all import vector, QQ
        sage: decoding = {0: 'a', 1: 'b', 2: 'c'}
        sage: v = vector(QQ, [2, -3, 1])
        sage: vector_to_element(decoding, v)
        [(2, 'a'), (-3, 'b'), (1, 'c')]
    """
    return [(vec[i],decoding[i]) for i in range(len(vec)) if (not sparse) or (vec[i] != 0)]
    
def operator_to_matrix(op, codomain_encoding, ins, base_ring=None):
    r"""
    Given a (linear) function ``op`` that takes elements of ``ins`` to another vector space, presented as a list of tuples (coefficient, base) encode the result as a matrix with the output encoded by ``codomain_encoding``.

    EXAMPLES:
    
    Example 1: Squaring operator on polynomials::

        sage: op = lambda z : list(z^2)
        sage: A.<x1,x2> = QQ['x1,x2']
        sage: encoding = {x1^2:0, x1*x2:1, x2^2:2}
        sage: operator_to_matrix(op, encoding, [x1,x2])
        [1 0]
        [0 0]
        [0 1]

    Example 2: Divided difference operator:: 

        sage: from schubert_polynomials import divided_difference_on_monomial_closed
        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: domain_basis = [x1^2, x1*x2, x2^2]
        sage: codomain_basis = [x1, x2]
        sage: codomain_encoding = {x1: 0, x2: 1}
        sage: dd_op = lambda mon:  list(divided_difference_on_monomial_closed(1, mon))
        sage: operator_to_matrix(dd_op, codomain_encoding, domain_basis, base_ring=QQ)
        [ 1  0 -1]
        [ 1  0 -1]

    Example 3: CombinatorialFreeModule with a projection operator::

        sage: from sage.all import CombinatorialFreeModule
        sage: F = CombinatorialFreeModule(QQ, ['a', 'b', 'c'])
        sage: a, b, c = F.basis()['a'], F.basis()['b'], F.basis()['c']
        sage: # Define a projection operator that kills 'c' component
        sage: def proj_op(elem):
        ....:     result = [(coeff, key) for (key, coeff) in elem if key in ['a', 'b']]
        ....:     return result
        sage: domain_basis = [a, b, c]
        sage: codomain_encoding = {'a': 0, 'b': 1}
        sage: operator_to_matrix(proj_op, codomain_encoding, domain_basis, base_ring=QQ)
        [1 0 0]
        [0 1 0]

    Example 4: CombinatorialFreeModule with a swap operator::

        sage: G = CombinatorialFreeModule(QQ, ['x', 'y', 'z'])
        sage: x, y, z = G.basis()['x'], G.basis()['y'], G.basis()['z']
        sage: # Operator that swaps x and y, and zeros out z
        sage: def swap_op(elem):
        ....:     result = []
        ....:     for (key, coeff) in elem:
        ....:         if key == 'x':
        ....:              result.append((coeff, 'y'))
        ....:         elif key == 'y': 
        ....:             result.append((coeff, 'x'))
        ....:     return result
        sage: domain_basis = [x, y, z]
        sage: codomain_encoding = {'x': 0, 'y': 1, 'z': 2}
        sage: operator_to_matrix(swap_op, codomain_encoding, domain_basis, base_ring=QQ)
        [0 1 0]
        [1 0 0]
        [0 0 0]

    Example 5: CombinatorialFreeModule with scaling operator::

        sage: H = CombinatorialFreeModule(QQ, [1, 2, 3])
        sage: b1, b2, b3 = H.basis()[1], H.basis()[2], H.basis()[3]
        sage: # Operator that scales each basis element by its index
        sage: def scale_op(elem):
        ....:     return [(coeff * key, key) for (key, coeff) in elem]
        sage: domain_basis = [b1, b2, b3]
        sage: codomain_encoding = {1: 0, 2: 1, 3: 2}
        sage: operator_to_matrix(scale_op, codomain_encoding, domain_basis, base_ring=QQ)
        [1 0 0]
        [0 2 0]
        [0 0 3]

    Example 6: CombinatorialFreeModule to polynomial ring::

        sage: F = CombinatorialFreeModule(QQ, [(2,0), (1,1), (0,2)])
        sage: R.<x,y> = QQ['x,y']
        sage:  # Operator that interprets tuple (a,b) as monomial x^a*y^b
        sage: def to_poly_op(elem):
        ....:     result = []
        ....:     for (key, coeff) in elem:
        ....:         mon = x^key[0] * y^key[1]
        ....:         result. append((coeff, mon))
        ....:     return result
        sage: domain = [F.basis()[(2,0)], F.basis()[(1,1)], F.basis()[(0,2)]]
        sage: codomain_encoding = {x^2: 0, x*y: 1, y^2: 2}
        sage: operator_to_matrix(to_poly_op, codomain_encoding, domain, base_ring=QQ)
        [1 0 0]
        [0 1 0]
        [0 0 1]

    Example 7: Working with list form for multi-symmetric functions::

        sage: from sage.all import SymmetricFunctions
        sage: s = SymmetricFunctions(QQ).s()
        sage: # Elements in "list form" as lists of (coefficient, support) tuples
        sage: input1 = [(1, (s[2], s[1]))]
        sage: input2 = [(1, (s[1,1], s[2]))]
        sage: # Define operator on list form
        sage: def apply_omega_first_coord(elem_list):
        ....:     result = []
        ....:     for (coeff, tup) in elem_list:
        ....:         new_tup = (tup[0].omega(), tup[1])
        ....:         result.append((coeff, new_tup))
        ....:     return result
        sage: inputs = [input1, input2]
        sage: codomain_encoding = {(s[1,1], s[1]): 0, (s[2], s[2]): 1}
        sage: operator_to_matrix(apply_omega_first_coord, codomain_encoding, inputs, base_ring=QQ)
        [1 0]
        [0 1]
    """
    if ins:
        image = [element_to_vector(codomain_encoding, op(elm), base_ring=base_ring) for elm in ins]
        return matrix(image).transpose()
    elif base_ring:
        warnings.warn("Input " + str(ins) + " is empty, producing an empty matrix!")
        matrix(base_ring,[])
    else:
        raise ValueError("Both input list and base_ring are undefined; cannot produce empty matrix!")


def operator_from_matrix(mat, domain_encoding, codomain_decoding):
    r"""
    Construct a linear operator from its matrix representation.

    Given a matrix and encodings for the domain and codomain, return a function that 
    applies the linear operator to elements represented as lists of (coefficient, basis_element) tuples.

    This is the inverse operation to ``operator_to_matrix``: it reconstructs the operator
    from its matrix form.

    EXAMPLES::

        sage: from sage.all import matrix, QQ
        sage: # Define a simple 2x2 matrix representing a linear operator
        sage: M = matrix(QQ, [[1, 2], [3, 4]])
        sage: domain_encoding = {'x': 0, 'y': 1}
        sage: codomain_decoding = {0: 'u', 1: 'v'}
        sage: op = operator_from_matrix(M, domain_encoding, codomain_decoding)
        sage: # Apply to element 1*x + 0*y
        sage: op([(1, 'x'), (0, 'y')])
        [(1, 'u'), (3, 'v')]
        sage: # Apply to element 0*x + 1*y
        sage: op([(0, 'x'), (1, 'y')])
        [(2, 'u'), (4, 'v')]
        sage: # Apply to element 2*x + 3*y
        sage: op([(2, 'x'), (3, 'y')])
        [(8, 'u'), (18, 'v')]

    Example with polynomial monomials::

        sage: R.<x,y> = QQ['x,y']
        sage: M = matrix(QQ, [[2, 1, 0], [0, 0, 0]])
        sage: domain_encoding = {x^2: 0, x*y: 1, y^2: 2}
        sage: codomain_decoding = {0: x, 1: y}
        sage: partial_x = operator_from_matrix(M, domain_encoding, codomain_decoding)
        sage: # Apply to x^2
        sage: partial_x([(1, x^2)])
        [(2, x), (0, y)]
        sage: # Apply to xy
        sage: partial_x([(1, x*y)])
        [(1, x), (0, y)]
        sage: # Apply to y^2
        sage: partial_x([(1, y^2)])
        [(0, x), (0, y)]

    Example with CombinatorialFreeModule:: 

        sage: from sage.all import CombinatorialFreeModule
        sage: F = CombinatorialFreeModule(QQ, ['a', 'b', 'c'])
        sage: # Projection matrix that kills 'c' component
        sage: M = matrix(QQ, [[1, 0, 0], [0, 1, 0]])
        sage: domain_encoding = {'a': 0, 'b': 1, 'c':  2}
        sage: codomain_decoding = {0: 'a', 1: 'b'}
        sage: proj = operator_from_matrix(M, domain_encoding, codomain_decoding)
        sage: # Apply to a + 2b + 3c
        sage: proj([(1, 'a'), (2, 'b'), (3, 'c')])
        [(1, 'a'), (2, 'b')]

    Example:  Round-trip with operator_to_matrix::

        sage: R.<x1,x2> = QQ['x1,x2']
        sage: # Original operator:  squaring
        sage: square_op = lambda z: list(z^2)
        sage: domain_basis = [x1, x2]
        sage: codomain_basis = [x1^2, x1*x2, x2^2]
        sage: codomain_encoding = {x1^2: 0, x1*x2: 1, x2^2: 2}
        sage: codomain_decoding = {0: x1^2, 1: x1*x2, 2: x2^2}
        sage: domain_encoding = {x1: 0, x2: 1}
        sage: # Convert operator to matrix
        sage: M = operator_to_matrix(square_op, codomain_encoding, domain_basis, base_ring=QQ)
        sage: M
        [1 0]
        [0 0]
        [0 1]
        sage: # Reconstruct operator from matrix
        sage: reconstructed_op = operator_from_matrix(M, domain_encoding, codomain_decoding)
        sage: # Test:  apply to x1
        sage: reconstructed_op([(1, x1)])
        [(1, x1^2), (0, x1*x2), (0, x2^2)]
        sage: # Test: apply to x2
        sage: reconstructed_op([(1, x2)])
        [(0, x1^2), (0, x1*x2), (1, x2^2)]

    Example with divided difference operator::

        sage: from schubert_polynomials import divided_difference_on_monomial_closed
        sage: R.<x1,x2,x3> = QQ['x1,x2,x3']
        sage: # Create matrix for divided difference operator
        sage: domain_basis = [x1^2, x1*x2, x2^2]
        sage: codomain_basis = [x1, x2]
        sage: domain_encoding = {x1^2: 0, x1*x2: 1, x2^2: 2}
        sage: codomain_encoding = {x1: 0, x2: 1}
        sage: codomain_decoding = {0: x1, 1: x2}
        sage: dd_op_original = lambda mon: list(divided_difference_on_monomial_closed(1, mon))
        sage: M = operator_to_matrix(dd_op_original, codomain_encoding, domain_basis, base_ring=QQ)
        sage: # Reconstruct the operator
        sage: dd_op_reconstructed = operator_from_matrix(M, domain_encoding, codomain_decoding)
        sage: # Test on x1^2
        sage: dd_op_reconstructed([(1, x1^2)])
        [(1, x1), (1, x2)]

    Example:  Composition of operators via matrix multiplication::

        sage: # Define two operators via matrices
        sage: M1 = matrix(QQ, [[1, 0], [0, 2]])  # Scaling operator
        sage: M2 = matrix(QQ, [[0, 1], [1, 0]])  # Swap operator
        sage: encoding = {'x': 0, 'y': 1}
        sage: decoding = {0: 'x', 1: 'y'}
        sage: op1 = operator_from_matrix(M1, encoding, decoding)
        sage: op2 = operator_from_matrix(M2, encoding, decoding)
        sage: # Compose via matrix multiplication
        sage: M_composed = M2 * M1
        sage: op_composed = operator_from_matrix(M_composed, encoding, decoding)
        sage: # Test:  should swap then scale
        sage: op_composed([(3, 'x'), (4, 'y')])
        [(8, 'x'), (3, 'y')]
    """
    return lambda inp : vector_to_element(codomain_decoding,mat*element_to_vector(domain_encoding, inp))
