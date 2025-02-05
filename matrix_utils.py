from sage.all import MatrixSpace, QQ, vector, block_matrix
import bisect
from itertools import chain
from functools import reduce

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
