
# Basic modules
from fatDAE.base.basic_import import *

class solver_ls:

    def __init__(self):
        pass

    def solve(self, A, b):
        pass

class solver_sp(solver_ls):

    def __init__(self):
        solver_ls.__init__(self)

    def solve(self, A, b):
        return scipy.sparse.linalg.spsolve(A, b)

class solver_lq(solver_ls):

    def __init__(self):
        solver_ls.__init__(self)

    def solve(self, A, b):
        return scipy.sparse.linalg.lsqr(A, b, atol=1e-14, btol=1e-14)[0]
