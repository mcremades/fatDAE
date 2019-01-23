# Date: 23/06/2018
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'..'); from fatDAE.base.basic_import import *

# User defined
import fatDAE.class_butcher
import fatDAE.class_solvers

class Problem(object):
    ''' Initial value problem goberned by a quasi-linearly implicit differential-algebraic system.

    .. math::
        \\begin{split}
            M(t, \\mathbf{x})\\frac{d\\mathbf{x}}{dt}(t) & = \mathbf{f}(t, \\mathbf{x}(t)),\quad t_0\le t\leq t_f \\\\
                          \\mathbf{x}(t_0) & = \\mathbf{x}_0
        \\end{split}

    Args:
        M (:obj:`function`)
        f (:obj:`function`)
        x_0 (:obj:`numpy.ndarray`)
        t_0 (:obj:`int`)
        t_f (:obj:`int`)
        derivatives (:obj:`dict`, optional)

    Attributes:
        M (:obj:`function`)
        f (:obj:`function`)
        x_0 (:obj:`numpy.ndarray`)
        t_0 (:obj:`int`)
        t_f (:obj:`int`)
        dim (:obj:`int`)
        dMdx (:obj:`function`, optional)
        dMdt (:obj:`function`, optional)
        dfdx (:obj:`function`, optional)
        dfdt (:obj:`function`, optional)
        t_list (:obj:`list`)
        x_list (:obj:`list`)

    .. note::
        If the matrix

        .. math::
            \\begin{equation}
                M(t,\\mathbf{x})\\in \mathbb{R}^{d\\times d}
            \\end{equation}

        is constant, then :attr:`M` must be a :obj:`scipy.sparse.csc_matrix` instead of a :obj:`function` in order to save matrix evaluations.
    '''

    def update(self, M, f, x_0, t_0, t_f, derivatives={}):

        self.M = M
        self.f = f

        self.x_0 = x_0
        self.t_0 = t_0
        self.t_f = t_f

        self.dim = len(self.x_0)

        if 'dMdx' in derivatives:
            self.dMdx = derivatives['dMdx']
        else:
            self.dMdx = None
        if 'dMdt' in derivatives:
            self.dMdt = derivatives['dMdt']
        else:
            self.dMdt = None

        if 'dfdx' in derivatives:
            self.dfdx = derivatives['dfdx']
        else:
            self.dfdx = None
        if 'dfdt' in derivatives:
            self.dfdt = derivatives['dfdt']
        else:
            self.dfdt = None

    def __init__(self, M, f, x_0, t_0, t_f, derivatives={}):

        self.M = M
        self.f = f

        self.x_0 = x_0
        self.t_0 = t_0
        self.t_f = t_f

        self.dim = len(self.x_0)

        if 'dMdx' in derivatives:
            self.dMdx = derivatives['dMdx']
        else:
            self.dMdx = None
        if 'dMdt' in derivatives:
            self.dMdt = derivatives['dMdt']
        else:
            self.dMdt = None

        if 'dfdx' in derivatives:
            self.dfdx = derivatives['dfdx']
        else:
            self.dfdx = None
        if 'dfdt' in derivatives:
            self.dfdt = derivatives['dfdt']
        else:
            self.dfdt = None

        self.t_list = []
        self.x_list = []

        self.delta_x_list = []
        self.delta_y_list = []

        self.err_est_list = []
        self.err_exc_list = []

    def solve(self, solver, state_machine=None, h=None, adp=False, adj=False, tlm=False):
        ''' Solves with a given solver the problem.

        Args:
            solver (:obj:`class_solvers.Solver`): Solver instance.
            adp (:obj:`bool`):
            adj (:obj:`bool`):
        '''

        if h == None:
            h = (self.t_f - self.t_0) / 1000.

        if adj:
            return solver.solve_adj(self, state_machine, h, adp, tlm)
        else:

            if adp:
                return solver.solve_adp(self, state_machine, h, adj, tlm)
            else:
                return solver.solve_fxd(self, state_machine, h, adj, tlm)

    def error(self, x, y, a_tol=1e-2, r_tol=1e-2):
        ''' Computes the error between two states.

        In particular, the error is computed as:

        .. math::
            \\begin{equation}
                error = \\frac{||x-y||_2}{a_{tol} + r_{tol} * \\max(||x||_2,||y||_2)}
            \\end{equation}

        Args:
            x (:obj:`numpy.ndarray`): State.
            y (:obj:`numpy.ndarray`): State.
            a_tol (:obj:`float`, optional): Absolute tolerance.
            r_tol (:obj:`float`, optional): Relative tolerance.

        Returns:
            (:obj:`float`): Error between the two states.
        '''

        return (numpy.linalg.norm(x - y, ord=numpy.inf)) / (a_tol + r_tol * max(numpy.linalg.norm(x, ord=numpy.inf), numpy.linalg.norm(y, ord=numpy.inf)))

    def store(self, t, x, delta_x=None, delta_y=None):
        ''' Stores a given time and state.

        Args:
            t (:obj:`float`): Time.
            x (:obj:`numpy.ndarray`): State.
        '''

        self.t_list.append(t)
        self.x_list.append(x)

        if delta_x == None:
            pass
        else:
            self.delta_x_list.append(delta_x)

        if delta_y == None:
            pass
        else:
            self.delta_y_list.append(delta_y)

        if hasattr(self, 'x_exact'):
            self.err_exc_list.append(self.error(x, self.x_exact(t)))

    def clean(self):
        ''' Erases stored times and states.
        '''

        self.t_list = []
        self.x_list = []

        self.delta_x_list = []
        self.delta_y_list = []

        self.err_est_list = []
        self.err_exc_list = []

    def check_order(self, solver, nsteps=8):
        ''' Check the convergence order of a given solver.

        Args:
            solver (:obj:`class_solver.Solver`): Solver instance.
            nsteps (:obj:`int`)
        '''

        if hasattr(self, 'exact') == False:
            raise NameError('To compute error an exact solution must be given...')

        h_list = []
        e_list = []

        for i in range(nsteps):
            h_list.append(1.0 / (10 * 2 ** i))

        o_1 = []
        o_2 = []
        o_3 = []

        for i in range(nsteps):

            self.solve(solver, h_list[i])

            e = 0.0

            for err in self.err_exc_list:
                e = e + err ** 2

            e_list.append(numpy.sqrt(e / len(self.err_exc_list)))

            print('Step: ', h_list[i], 'Error: ', e_list[i])

            if i > 0:
                print("Convergence Rate: ", numpy.log(e_list[-1] / e_list[-2]) / numpy.log(h_list[i] / h_list[i - 1]))

            o_1.append(h_list[i] ** 1.0)
            o_2.append(h_list[i] ** 2.0)
            o_3.append(h_list[i] ** 3.0)

        matplotlib.pyplot.figure()

        matplotlib.pyplot.plot(h_list, o_1, 'b--', label = "$O(h^1)$")
        matplotlib.pyplot.plot(h_list, o_2, 'r--', label = "$O(h^2)$")
        matplotlib.pyplot.plot(h_list, o_3, 'g--', label = "$O(h^3)$")

        if solver.advancing_table.p == 1:
            x_color = 'b'
        else:
            if solver.advancing_table.p == 2:
                x_color = 'r'
            else:
                if solver.advancing_table.p == 3:
                    x_color = 'g'
                else:
                    raise NameError('Expecting lower order methods...')

        matplotlib.pyplot.plot(h_list, e_list, x_color)

        matplotlib.pyplot.xscale('log')
        matplotlib.pyplot.yscale('log')

        matplotlib.pyplot.xlabel('Steps')
        matplotlib.pyplot.ylabel('Error')

        matplotlib.pyplot.legend(loc=0)

        matplotlib.pyplot.title(str(solver.advancing_table.name))

        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()

    def get_delta_x(self, i, j):

        delta_x_list = []

        for delta_x in self.delta_x_list:
            delta_x_list.append(delta_x[i, j])

        return delta_x_list

    def get_delta_y(self, i, j):

        delta_y_list = []

        for delta_y in self.delta_y_list:
            delta_y_list.append(delta_y[i, j])

        return delta_y_list

    def plot(self):
        ''' Plot stored times versus stored states.
        '''

        nrows = len(self.x_list[0])
        ncols = len(self.x_list[0])

        print(self.t_list)

        image = []

        for i in range(len(self.x_list[0])):

            for j in range(len(self.x_list[0])):

                image.append(numpy.trapz(self.get_delta_x(i, j), x=self.t_list) / (self.t_list[-1]))

        image = numpy.array(image); image = image.reshape((nrows, ncols))

        row_labels = []
        col_labels = []

        for i in range(len(self.x_list[0])):
            row_labels.append('$x_{'+str(i)+'}$')
            col_labels.append('$x_{'+str(i)+'}$')

        matplotlib.pyplot.matshow(image)
        matplotlib.pyplot.xticks(range(len(self.x_list[0])), col_labels)
        matplotlib.pyplot.yticks(range(len(self.x_list[0])), row_labels)

        matplotlib.pyplot.colorbar()

        matplotlib.pyplot.show()

        if len(self.delta_x_list) > 0:

            figure, axes = matplotlib.pyplot.subplots(len(self.x_list[0]), len(self.x_list[0]))

            for i in range(len(self.x_list[0])):
                for j in range(len(self.x_list[0])):

                    axes[i, j].plot(self.t_list, self.get_delta_x(i, j))

                    axes[i, j].set_title('$\delta_{'+str(i)+str(j)+'}$')
                    axes[i, j].set_xlabel('$t$')
                    axes[i, j].grid()

            matplotlib.pyplot.subplots_adjust(wspace=0.4, hspace=0.6)

        if len(self.x_list[0]) < 6:

            for i in range(len(self.x_list[0])):

                x_calc_list = []
                x_exac_list = []

                if hasattr(self, 'exact'):

                    for t in self.t_list:
                        x_exac_list.append(self.exact(t)[i])

                    for x in self.x_list:
                        x_calc_list.append(x[i])
                else:
                    for x in self.x_list:
                        x_calc_list.append(x[i])

                matplotlib.pyplot.figure()

                if hasattr(self, 'exact'):
                    matplotlib.pyplot.plot(self.t_list, x_calc_list, 'b-', label='Exac.')
                    matplotlib.pyplot.plot(self.t_list, x_exac_list, 'rx', label='Calc.')
                else:
                    matplotlib.pyplot.plot(self.t_list, x_calc_list, 'b-')

                matplotlib.pyplot.xlabel('$t$')
                matplotlib.pyplot.ylabel('$x_'+str(i)+'$')

                matplotlib.pyplot.grid()
        else:

            matplotlib.pyplot.figure()

            matplotlib.pyplot.plot(self.t_list, self.x_list)

            matplotlib.pyplot.xlabel('Times')
            matplotlib.pyplot.ylabel('State')

            matplotlib.pyplot.grid()

        matplotlib.pyplot.show()


class Control(Problem):
    '''Optimal control problem governed by a quasi-linearly implicit differential-algebraic system.

    .. math::
        \\begin{split}
            \\text{minimize} \\quad \\Psi(\\mathbf{u}) & = J(t_f, \\mathbf{x}(t_f), \\mathbf{u}) + \\int_{t_0}^{t_f}g(t, \\mathbf{x}(t), \\mathbf{u})\\; dt \\\\
            \\text{s.t.} \\quad \\quad \\quad \\quad & \\\\
            M(t, \\mathbf{x})\\frac{d\\mathbf{x}}{dt}(t) & = \mathbf{f}(t, \\mathbf{x}(t), \\mathbf{u}),\quad t_0\le t\leq t_f \\\\
                          \\mathbf{x}(t_0) & = \\mathbf{x}_0
        \\end{split}

    .. warning::
        Computation of gradients for cost functionals with an explicit control dependence with the adjoint method are not currently working, only an approximation is computed.

    Args:
        M (:obj:`function`)
        f (:obj:`function`)
        x_0 (:obj:`numpy.ndarray`)
        t_0 (:obj:`int`)
        t_f (:obj:`int`)
        control (:obj:`runge_kutta.class_control.Control`)
        J (:obj:`function`)
        g (:obj:`function`)
        derivatives (:obj:`dict`, optional)

    Attributes:
        J (:obj:`function`)
        g (:obj:`function`)
        dJdx (:obj:`function`, optional)
        dJdu (:obj:`function`, optional)
        dgdx (:obj:`function`, optional)
        dgdu (:obj:`function`, optional)
        dfdu (:obj:`function`, optional)
        d2fdxdx (:obj:`function`, optional)
        d2fdxdt (:obj:`function`, optional)
        d2fdxdu (:obj:`function`, optional)
        d2fdtdu (:obj:`function`, optional)
    '''
    def __init__(self, M, f, x_0, t_0, t_f, J=None, g=None, derivatives={}):

        Problem.__init__(self, M, f, x_0, t_0, t_f, derivatives)

        self.J = J

        if 'dJdx' in derivatives:
            self.dJdx = derivatives['dJdx']
        else:
            self.dJdx = None
        if 'dJdu' in derivatives:
            self.dJdu = derivatives['dJdu']
        else:
            self.dJdu = None

        self.g = g

        if 'dgdx' in derivatives:
            self.dgdx = derivatives['dgdx']
        else:
            self.dgdx = None
        if 'dgdu' in derivatives:
            self.dgdu = derivatives['dgdu']
        else:
            self.dgdu = None

        if 'dfdu' in derivatives:
            self.dfdu = derivatives['dfdu']
        else:
            self.dfdu = None

        if 'dMdu' in derivatives:
            self.dMdu = derivatives['dMdu']
        else:
            self.dMdu = None

        if 'd2fdxdx' in derivatives:
            self.d2fdxdx = derivatives['d2fdxdx']
        else:
            self.d2fdxdx = None
        if 'd2fdxdt' in derivatives:
            self.d2fdxdt = derivatives['d2fdxdt']
        else:
            self.d2fdxdt = None
        if 'd2fdxdu' in derivatives:
            self.d2fdxdu = derivatives['d2fdxdu']
        else:
            self.d2fdxdu = None

        if 'd2fdtdu' in derivatives:
            self.d2fdtdu = derivatives['d2fdtdu']
        else:
            self.d2fdtdu = None

class Fitting(Control):

    def __init__(self, M, f, x_0, t_0, t_f, t_obs, x_obs, h, dhdx, dhdu, derivatives={}):

        self.t_obs = t_obs
        self.x_obs = x_obs

        self.h = h

        self.dhdx = dhdx
        self.dhdu = dhdu

        self.x_int = scipy.interpolate.interp1d(self.t_obs, \
                                                self.x_obs, fill_value="extrapolate")

        def g(t, x):
            return (self.h(t, x) - self.x_int(t)) ** 2

        def dgdx(t, x):
            return 2. * (self.h(t, x) - self.x_int(t)) * self.dhdx(t, x)

        def dgdu(t, x):
            return 2. * (self.h(t, x) - self.x_int(t)) * self.dhdu(t, x)

        derivatives['dgdx'] = dgdx
        derivatives['dgdu'] = dgdu
        derivatives['dJdx'] = dgdx
        derivatives['dJdu'] = dgdu

        Control.__init__(self, M, f, x_0, t_0, t_f, J=g, g=g, derivatives=derivatives)

    def plot(self):

        matplotlib.pyplot.figure()

        self.t_clc = []
        self.x_clc = []

        for t, x in zip(self.t_list, self.x_list):

            self.t_clc.append(t); self.x_clc.append(self.h(t, x))

        matplotlib.pyplot.plot(self.t_clc, self.x_clc, 'b-')
        matplotlib.pyplot.plot(self.t_obs, self.x_obs, 'bo')

        matplotlib.pyplot.xlabel('Times')
        matplotlib.pyplot.ylabel('Output')

        matplotlib.pyplot.grid()
        matplotlib.pyplot.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('json', help = "Butcher's tables")

    parser.add_argument('-adv', help = "Advancing method, choose 1 or 2.", action = 'store', required = True)
    parser.add_argument('-est', help = "Estimator method, choose 1 or 2.", action = 'store')

    args = parser.parse_args()

    with open(args.json) as data_file:
        butcher_json = json.load(data_file)

    if args.adv == '1':
        embedded_1 = False
        embedded_2 = True
    else:
        if args.adv == '2':
            embedded_1 = True
            embedded_2 = False
        else:
            raise NameError('Choose between 1 or 2 for advancing method...')

    solver = fatDAE.class_solvers.build(butcher_json, embedded_1, embedded_2)

    '''
        Dahlquist problem

            y_1'(t) = - lmb * y_1(t)
            y_2'(t) = - dlt * y_2(t)

        with initial condition:

            y_1(0) = 1.
            y_2(0) = 1.

    '''

    lmb = 1.
    dlt = 5.

    def M(t, x):

        A = numpy.array([[1., 0.], \
                         [0., 1.]])

        return scipy.sparse.csc_matrix(A)

    def f(t, x):

        b = numpy.array([- lmb * x[0] - lmb * x[1], \
                         - dlt * x[1]])

        return b

    def dfdx(t, x):

        A = numpy.array([[- lmb, -lmb], \
                         [0., -dlt]])

        return scipy.sparse.csc_matrix(A)

    def dfdt(t, x):

        b = numpy.array([0., \
                         0.])

        return b

    x_0 = numpy.array([1., \
                       2.])

    t_0 = 0.
    t_f = 1.

    derivatives = {'dfdx': dfdx, \
                   'dfdt': dfdt}

    problem = Problem(M(t_0, x_0), f, x_0, t_0, t_f, derivatives)

    def x(t):

        b = numpy.array([x_0[0] * numpy.exp(-lmb * t), \
                         x_0[1] * numpy.exp(-dlt * t)])

        return b

    problem.exact = x

    problem.solve(solver, h=5.e-2, tlm=True); problem.plot()
