# Date: 24/07/2017
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'../..'); from fatDAE.base.basic_import import *

# Dolfin package
from dolfin import *

class Control(object):
    ''' Time dependent control.

    Args:
        t_list (:obj:`list`)
        t_vrbl (:obj:`list`)

    Attributes:
        t_list (:obj:`list`)
        u_list (:obj:`list`)
        u (:obj:`ufl.algebra.Sum`)
    '''

    def __init__(self, t_list, t_vrbl):

        self.t_list = t_list
        self.t_vrbl = t_vrbl

    def build_list(self, u_list):
        ''' Build a list of controls.

        Args:
            u_list(:obj:`list`)
        '''

        self.u_list = u_list

    def build_expr(self, u_expr, f_spce):
        ''' Build a list of controls by interpolating and expression into a function space.

        Args:
            u_expr(:obj:`dolfin.functions.expression.CompiledExpression`)
            f_spce(:obj:`dolfin.functions.functionspace.FunctionSpace`)
        '''

        self.u_list = []

        for i in range(len(self.t_list)):

            u_expr.t = self.t_list[i]; self.u_list.append(interpolate(u_expr, f_spce))

class Bernstein(Control):
    ''' Time dependent control using Bernstein interpolation.

    Args:
        t_list (:obj:`list`)
        t_vrbl (:obj:`list`)

    Attributes:
        t_list (:obj:`list`)
        u_list (:obj:`list`)
        u (:obj:`ufl.algebra.Sum`)
    '''

    def __init__(self, t_list, t_vrbl):

        Control.__init__(self, t_list, t_vrbl)

    def build_ctrl(self):
        ''' Interpolates the control using Bernstein basis functions.

        .. math::
            \\begin{equation}
                u(t) = \\sum_{i=0}^{n} u_i b_i(t)
            \\end{equation}

        were we have introduced

        .. math::
            \\begin{equation}
                b_i(t) = {{i}\choose{n}} t^i (1 - t) ^ {(n - i)},\\quad i=1,\\dots, n
            \\end{equation}

        known as Bernstein polynomials.

        .. warning::
            This interpolation only works if the control is defined within the unit interval...
        '''

        self.u = 0.0

        for i in range(len(self.t_list)):

            self.u = self.u \
                   + self.u_list[i] * scipy.special.binom(len(self.t_list), i) * self.t_vrbl ** i * (1.0 - self.t_vrbl) ** (len(self.t_list) - i)

class Lagrange(Control):
    ''' Time dependent control using Lagrange interpolation.

    Args:
        t_list (:obj:`list`)
        t_vrbl (:obj:`list`)

    Attributes:
        t_list (:obj:`list`)
        u_list (:obj:`list`)
        u (:obj:`ufl.algebra.Sum`)
        chi_list (:obj:`list`)
        phi_list (:obj:`list`)
    '''

    def __init__(self, t_list, t_vrbl):

        Control.__init__(self, t_list, t_vrbl)

        self.build_chi()

    def build_chi(self):
        ''' Build the characteristic functions.

        They take the form:

        .. math::
            \\begin{equation}
                \\chi_i(t) = \\{
                \\begin{array}{cll}
                    1 & if & t_i < t \\leq  t_{i+1}\\\\
                    0 & otherwise &
                \\end{array}
            \\end{equation}

        for :math:`i=1,\\dots, n-1` and

        .. math::
            \\begin{equation}
                \\chi_0(t) = \\{
                \\begin{array}{cll}
                    1 & if & t_0 \\leq t \\leq  t_{1}\\\\
                    0 & otherwise &
                \\end{array}
            \\end{equation}

        and store them in :attr:`chi_list`.
        '''

        self.chi_list = []

        def make_chi(j):

            if j == (len(self.t_list) - 2):

                def chi():
                    return conditional(ge(self.t_vrbl, self.t_list[j]), conditional(le(self.t_vrbl, self.t_list[j + 1]), 1.0, 0.0), 0.0)

            else:

                def chi():
                    return conditional(ge(self.t_vrbl, self.t_list[j]), conditional(lt(self.t_vrbl, self.t_list[j + 1]), 1.0, 0.0), 0.0)

            return chi

        for j in range(len(self.t_list) - 1):

            self.chi_list.append(make_chi(j))

    def build_phi(self):
        ''' Build the basis functions.
        '''
        pass

    def build_ctrl(self):
        ''' Interpolates the control using Lagrange basis functions.

        .. math::
            \\begin{equation}
                u(t) = \\sum_{i=1}^{n}u_i\\varphi_i(t)
            \\end{equation}
        '''

        self.u = 0.0

        for i in range(len(self.t_list)):

            self.u = self.u + self.u_list[i] * self.phi_list[i]()

class P0(Lagrange):
    ''' Time dependent control using Lagrange interpolation of 0-th order.

    Args:
        t_list (:obj:`list`)
        t_vrbl (:obj:`list`)

    Attributes:
        t_list (:obj:`list`)
        u_list (:obj:`list`)
        u (:obj:`ufl.algebra.Sum`)
        chi_list (:obj:`list`)
        phi_list (:obj:`list`)
    '''

    def __init__(self, t_list, t_vrbl):

        Lagrange.__init__(self, t_list, t_vrbl)

        self.build_phi()

    def build_phi(self):
        ''' Build the basis functions.
        '''

        self.phi_list = []

        for j in range(0, len(self.t_list) - 1):

            self.phi_list.append(self.chi_list[j])

        def phi():
            return conditional(ge(self.t_vrbl, self.t_list[- 1]), 1.0, 0.0)

        self.phi_list.append(phi)

class P1(Lagrange):
    ''' Time dependent control using Lagrange interpolation of 1-th order.

    Args:
        t_list (:obj:`list`)
        t_vrbl (:obj:`list`)

    Attributes:
        t_list (:obj:`list`)
        u_list (:obj:`list`)
        u (:obj:`ufl.algebra.Sum`)
        chi_list (:obj:`list`)
        phi_list (:obj:`list`)
    '''

    def __init__(self, t_list, t_vrbl):

        Lagrange.__init__(self, t_list, t_vrbl)

        self.build_phi()

    def build_phi(self):
        ''' Build the basis functions.
        '''

        self.phi_list = []

        def phi():
            return (self.t_list[1] - self.t_vrbl) / (self.t_list[1] - self.t_list[0]) * self.chi_list[0]()

        self.phi_list.append(phi)

        def make_phi(j):

            def phi():

                phi_l = (self.t_vrbl - self.t_list[j - 1]) \
                      / (self.t_list[j] - self.t_list[j - 1])

                phi_r = (self.t_list[j + 1] - self.t_vrbl) \
                      / (self.t_list[j + 1] - self.t_list[j])

                return  phi_l * self.chi_list[j - 1]() + phi_r * self.chi_list[j]()

            return phi

        for j in range(1, len(self.t_list) - 1):

            self.phi_list.append(make_phi(j))

        def phi():
            return (self.t_vrbl - self.t_list[- 2]) / (self.t_list[- 1] - self.t_list[- 2]) * self.chi_list[- 1]()

        self.phi_list.append(phi)

if __name__ == '__main__':

    mesh = UnitSquareMesh(15, 15)

    X = FunctionSpace(mesh, 'CG', 1)

    t_expr = Expression('value', value=0.0, degree=0); t_vrbl = variable(t_expr)

    t_list = [0.0]; t = 0.0

    for j in range(5):
        t = t + 0.2; t_list.append(t)

    Control_P0 = P0(t_list, t_vrbl)
    Control_P1 = P1(t_list, t_vrbl)

    P0_pvd = File('P0.pvd')
    P1_pvd = File('P1.pvd')

    P0_plot = Function(X)
    P1_plot = Function(X)

    Control_P0.build_expr(Expression('x[0] * (x[0] - 1.) * x[1] * (x[1] - 1.) * t', t=0.0, degree=0), X)
    Control_P1.build_expr(Expression('x[0] * (x[0] - 1.) * x[1] * (x[1] - 1.) * t', t=0.0, degree=0), X)

    Control_P0.build_ctrl()
    Control_P1.build_ctrl()

    t_list = [0.0]; t = 0.0

    for j in range(10):
        t = t + 0.1; t_list.append(t)

    for t in t_list:

        t_expr.value = t; plot(Control_P0.u); assign(P0_plot, project(Control_P0.u, X))
        P0_pvd << (P0_plot, t)

    for t in t_list:

        t_expr.value = t; plot(Control_P1.u); assign(P1_plot, project(Control_P1.u, X))
        P1_pvd << (P1_plot, t)
