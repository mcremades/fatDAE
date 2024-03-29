
# Basic modules
#from fatdae.base.basic_import import *

# User defined
import fatdae.base.sp_solvers

import numpy

class solver_nl:

    def __init__(self, m_ite=20, a_tol=1e-12, r_tol=1e-6):

        self.m_ite = m_ite

        self.a_tol = a_tol
        self.r_tol = r_tol

    def solve(self):
        pass

class solver_fp(solver_nl):

    def solve(self, F, x):
        pass

class solver_nt(solver_nl):

    def __init__(self, m_ite=50, a_tol=1e-8, r_tol=1e-8, simplified=False):

        solver_nl.__init__(self, m_ite=m_ite, a_tol=a_tol, r_tol=r_tol)

        self.solver = fatdae.base.sp_solvers.solver_sp()

        self.simplified = False

    def solve(self, F, J, x):

        self.converged = False; self.diverged = False

        for j in range(self.m_ite):



                if callable(J):
                    Delta = self.solver.solve(J(x), - F(x))
                else:
                    Delta = self.solver.solve(J, - F(x))

                error = numpy.linalg.norm(Delta,numpy.inf) / numpy.linalg.norm(x,numpy.inf)

                #print(error)

                if error < self.a_tol:

                    self.converged = True

                    #print "Converged - >", self.converged

                    return x, j
                else:
                    pass
                    #print error

                x = x + Delta

                error_old = error

        print("Converged - >", self.converged)

        return x, j
