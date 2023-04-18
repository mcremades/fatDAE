import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import fatdae.solvers
import fatdae.dolfin_interface.class_problem
import dolfin





'''
        Mixed Heat equation

          p frac{partial phi}{partial t} - div(q(phi)mathbf{u}) = f in Omega times (0, T]
                                         mathbf{u} - Nabla{phi} = 0 in Omega times (0, T]
                                                            phi = 0 in Gamma times (0, T]


 '''

mesh = UnitSquareMesh(10,10)

P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
P0 = VectorElement('DG', mesh.ufl_cell(), 0)

ME = MixedElement([P1, P0])

V = FunctionSpace(mesh, P1)
Q = FunctionSpace(mesh, P0)

X = FunctionSpace(mesh, ME)

dxdt = TrialFunction(X)
dpdt = split(dxdt)[0]
dudt = split(dxdt)[1]

x = Function(X)
p = split(x)[0]
u = split(x)[1]

y = TestFunction(X)
q = split(y)[0]
v = split(y)[1]

k_1 = 1
k_2 = 1

f = Constant(1.0)

F = k_1 * dpdt * q * dx - f * q * dx \
      - k_2 * inner(u, grad(q)) * dx \
      + k_2 * inner(u, v) * dx \
      - k_2 * inner(grad(p), v) * dx

x_0 = numpy.array(x.vector()[:])

t_0 = 0.0
t_f = 1.0

problem = UFL_Problem(F, x_0, t_0, t_f)


class HeatEquation(fatdae.dolfin_interface.class_problem.UFL_Problem):
    
    def __init__(self):

        data = dolfin.Expression("16*x[0]*(x[0]-1)*x[1]*(x[1]-1)*sin(pi*t)",t=0,degree=4)
        nu = dolfin.Constant(1e-5)

        mesh = dolfin.UnitSquareMesh(8, 8)
        V = dolfin.FunctionSpace(mesh, "CG", 1)

        dudt = dolfin.TrialFunction(V)
        self.u_1 = dolfin.Function(V)
        v = dolfin.TestFunction(V)

        f = dolfin.Function(V)
        F = dudt*v*dolfin.dx + nu*dolfin.inner(dolfin.grad(self.u_1),dolfin.grad(v))*dolfin.dx - f*v*dolfin.dx
        x_0=None
        t_0=0
        t_f=2
        super().__init__(F, x_0, t_0, t_f)



problem = HeatEquation()