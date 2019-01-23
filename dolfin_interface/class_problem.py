# Date: 24/07/2017
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'..'); from fatDAE.base.basic_import import *

# Dolfin package
from dolfin import *

# User defined
import fatDAE.class_problem

class UFL_Problem(fatDAE.class_problem.Problem):
    ''' Initial value problem.

    It takes a variational formulation, for example: Find :math:`u \in L^2((t_0,t_f];H_0^1(\Omega))` such that

    .. math::
        \\begin{equation}
            \\int_\\Omega p(u)\\frac{\\partial u}{\\partial t}v\\;d\\Omega - \\int_{\\Omega}q(u)\\nabla u \\cdot \\nabla v\\;d\\Omega - \\int_{\\Omega}fv\\;d\\Omega = 0,\\quad \\forall v\\in H_{0}^{1}(\\Omega)
        \\end{equation}

    (were the time derivative must be declared as a trial function) in order to extract:

        - a bilinear form :attr:`M_form`, defined as

        .. math::
            \\begin{equation}
                M(\\frac{\\partial u}{\\partial t}, v; u) = \\int_\\Omega p(u)\\frac{\\partial u}{\\partial t}v\\;d\\Omega
            \\end{equation}

        - a bilinear form :attr:`f_form`, defined as

        .. math::
            \\begin{equation}
                \\hspace{4cm} f(v; u) = \\int_{\\Omega}q(u)\\nabla u \\cdot \\nabla v\\;d\\Omega + \\int_{\\Omega}fv\\;d\\Omega
            \\end{equation}

    Args:

    Attributes:
        M_form (:obj:`ufl.form.Form`): 2-form associated to the time derivative.
        f_form (:obj:`ufl.form.Form`): 1-form associated to the right hand side.
        I_interior (:obj:`scipy.sparse.csr_matrix`): Diagonal matrix with 1's in the diagonal for interior nodes and 0's otherwise, created by :meth:`set_boundary`.
        I_boundary (:obj:`scipy.sparse.csr_matrix`): Diagonal matrix with 0's in the diagonal for interior nodes and 1's otherwise, created by :meth:`set_boundary`.
        boundary_conditions (:obj:`list`): List of :obj:`dolfin.fem.bcs.DirichletBC` items, created by :meth:`set_boundary`.
        boundary_dervatives (:obj:`list`): List of :obj:`dolfin.fem.bcs.DirichletBC` items, created by :meth:`set_boundary`.
        dfdx_form: 2-form arising from the differentiation of :attr:`f_form`.
        dfdt_form: 1-form arising from the differentiation of :attr:`f_form`.

    .. warning::
        The time in the variational formulation...
    '''

    def update(self, t, x):

        self.M_form, self.f_form = lhs(self.F_var), rhs(self.F_var)

        def M(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.M_form)

        def f(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            f = assemble(self.f_form)

            for boundary_condition in self.boundary_conditions:
                boundary_condition.apply(f)

            return numpy.array(f)

        self.dfdx_form = derivative(self.f_form, self.x)

        def dfdx(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_dfdx(self.dfdx_form)

        self.dfdt_form = diff(self.f_form, self.t_v)

        def dfdt(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_dfdt(self.dfdt_form)

        self.dMdx_form = derivative(self.M_form, self.x, self.y)

        def dMdx(t, x, y):

            self.t.value = t

            self.x.vector()[:] = x
            self.y.vector()[:] = y

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.dMdx_form)

        self.dMdt_form = diff(self.M_form, self.t_v)

        def dMdt(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.dMdt_form)

        self.derivatives = {
                        'dMdx': dMdx,
                        'dMdt': dMdt,
                        'dfdx': dfdx,
                        'dfdt': dfdt
                        }

        # Matrix
        self.M = M

        # Source
        self.f = f

        self.t_0 = t
        self.x_0 = x

        # Matrix derivatives
        if 'dMdx' in self.derivatives:
            self.dMdx = derivatives['dMdx']
        else:
            self.dMdx = None
        if 'dMdt' in self.derivatives:
            self.dMdt = derivatives['dMdt']
        else:
            self.dMdt = None

        # Source derivatives
        if 'dfdx' in self.derivatives:
            self.dfdx = derivatives['dfdx']
        else:
            self.dfdx = None
        if 'dfdt' in self.derivatives:
            self.dfdt = derivatives['dfdt']
        else:
            self.dfdt = None

    def __init__(self, v_form, x_0, t_0, t_f):

        self.M_form, self.f_form = lhs(v_form), rhs(v_form)

        def M(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.M_form)

        def f(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            f = assemble(self.f_form)

            for boundary_condition in self.boundary_conditions:
                boundary_condition.apply(f)

            return numpy.array(f)

        self.dfdx_form = derivative(self.f_form, self.x)

        def dfdx(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_dfdx(self.dfdx_form)

        self.dfdt_form = diff(self.f_form, self.t_v)

        def dfdt(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_dfdt(self.dfdt_form)

        self.dMdx_form = derivative(self.M_form, self.x, self.y)

        def dMdx(t, x, y):

            self.t.value = t

            self.x.vector()[:] = x
            self.y.vector()[:] = y

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.dMdx_form)

        self.dMdt_form = diff(self.M_form, self.t_v)

        def dMdt(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.dMdt_form)

        self.derivatives = {
                        'dMdx': dMdx,
                        'dMdt': dMdt,
                        'dfdx': dfdx,
                        'dfdt': dfdt
                        }

        fatDAE.class_problem.Problem.__init__(self, M, f, x_0, t_0, t_f, self.derivatives)

    def assemble_csr(self, form):
        ''' Assembles a bilinear form.

        Args:
            form:

        Returns:
            :obj:`scipy.sparse.csr_matrix`

        '''

        M = PETScMatrix()

        M = assemble(form, tensor = M)

        M_csr = M.mat().getValuesCSR()

        M = scipy.sparse.csr_matrix((M_csr[2], M_csr[1], M_csr[0]), shape=(self.dim, self.dim))

        return M

    def assemble_M(self, form, bc=True):
        ''' Assembles a bilinear form and set to 0 the boundary nodes.

        Args:
            form:

        Returns:
            :obj:`scipy.sparse.csc_matrix`

        '''

        M = self.assemble_csr(form)

        if bc == True:

            # Dirichlet boundary conditions
            if hasattr(self, 'I_interior'):
                M = self.I_interior * M

        return scipy.sparse.csc_matrix(M)

    def assemble_f(self, form, bc=True):
        ''' Assembles a linear form and applies Dirichlet boundary conditions.

        Args:
            form:

        Returns:
            :obj:`numpy.ndarray`

        '''

        f = assemble(form)

        if bc == True:

            # Dirichlet boundary conditions
            for boundary_condition in self.boundary_conditions:
                boundary_condition.apply(f)

        return numpy.array(f)

    def assemble_dfdx(self, form):
        ''' Assembles a bilinear form and set to 1 the boundary nodes.

        Args:
            form:

        Returns:
            :obj:`scipy.sparse.csc_matrix`

        .. warning::
            Not suited for non-linear Dirichlet boundary conditions.

        '''

        dfdx = self.assemble_csr(form)

        # Dirichlet boundary conditions
        if hasattr(self, 'I_interior'):
            dfdx = self.I_interior * dfdx + self.I_boundary

        return scipy.sparse.csc_matrix(dfdx)

    def assemble_dfdt(self, form):
        ''' Assembles a linear form and applies Dirichlet boundary conditions time derivatives.

        Args:
            form:

        Returns:
            :obj:`numpy.ndarray`

        '''
        dfdt = assemble(form)

        # Dirichlet boundary conditions
        for boundary_derivative in self.boundary_dervatives:
            boundary_derivative.apply(dfdt)

        return numpy.array(dfdt)

    def set_boundary(self, var_list=[], exp_list=[], der_list=[], bnd_list=[]):
        ''' Build a list of Dirichlet boundary conditions from lists of variables, expressions, derivatives and boundaries.

        Args:
            var_list
            exp_list
            der_list
            bnd_list

        '''

        # List with Dirichlet boundary conditions
        self.boundary_conditions = []
        self.boundary_dervatives = []

        # Non linear treatment of Dirichlet boundary conditions
        nl_exp_list = []
        nl_der_list = []
        nl_fsp_list = []

        for i in range(len(exp_list)):

            if len(exp_list[i]) == 1:

                nl_fsp_list.append(var_list[i].function_space())

                nl_exp_list.append(Expression("u - g", degree=1, u = var_list[i], g = exp_list[i][0]))
                nl_der_list.append(Expression("0 - g", degree=1, u = var_list[i], g = der_list[i][0]))

            else:

                for j in range(len(exp_list[i])):

                    nl_fsp_list.append(var_list[i].function_space().sub(j))

                    nl_exp_list.append(Expression("u - g", degree=1, u = var_list[i].sub(j), g = exp_list[i][j]))
                    nl_der_list.append(Expression("0 - g", degree=1, u = var_list[i].sub(j), g = der_list[i][j]))

        for i in range(len(exp_list)):

            for j in range(len(exp_list[i])):

                self.boundary_conditions.append(DirichletBC(nl_fsp_list[i], nl_exp_list[i], bnd_list[i]))
                self.boundary_dervatives.append(DirichletBC(nl_fsp_list[i], nl_der_list[i], bnd_list[i]))

        # Matrices to apply Dirichlet boundary conditions
        chi_interior = numpy.ones(self.dim); chi_boundary = numpy.zeros(self.dim)

        for boundary_condition in self.boundary_conditions:
            for key, value in boundary_condition.get_boundary_values().items():
                chi_interior[key] = 0.0
                chi_boundary[key] = 1.0

        self.I_interior = scipy.sparse.spdiags(chi_interior, [0], self.dim, self.dim).tocsr()
        self.I_boundary = scipy.sparse.spdiags(chi_boundary, [0], self.dim, self.dim).tocsr()

class UFL_Control(fatDAE.class_problem.Control, UFL_Problem):
    '''Optimal control problem.

    Args:

    Attributes:

        J_form (:obj:`ufl.form.Form`): 0-form associated to the cost functional.
        dJdx_form (:obj:`ufl.form.Form`): 1-form associated to :attr:`J_form` derivatives with respect to the state.
        dJdu_form (:obj:`list`): List with 1-forms associated to :attr:`J_form` derivatives with respect to the control.
        g_form (:obj:`ufl.form.Form`): 0-form associated to the cost functional.
        dgdx_form (:obj:`ufl.form.Form`): 1-form associated to :attr:`g_form` derivatives with respect to the state.
        dgdu_form (:obj:`list`): List with 1-forms associated to :attr:`g_form` derivatives with respect to the control.
        d2fdxdx_form (:obj:`ufl.form.Form`): 2-form associated to :attr:`dolfin_interface.class_problem.Problem.dfdx_form` derivatives with respect to the state.
        d2fdxdt_form (:obj:`ufl.form.Form`): 2-form associated to :attr:`dolfin_interface.class_problem.Problem.dfdx_form` derivatives with respect to the times.
        d2fdxdu_form (:obj:`list`): List with 2-forms associated to :attr:`dolfin_interface.class_problem.Problem.dfdx_form` derivatives with respect to the control.
        d2fdtdu_form (:obj:`list`): List with 2-forms associated to :attr:`dolfin_interface.class_problem.Problem.dfdt_form` derivatives with respect to the control.


    '''
    def __init__(self, control, J_form, g_form, v_form, x_0, t_0, t_f):


        UFL_Problem.__init__(self, v_form, x_0, t_0, t_f)

        self.dMdu_form = []
        self.dfdu_form = []

        self.u_type = []

        for u in control:

            if type(u) == type(Constant(0.)):

                self.dMdu_form.append(diff(self.M_form, u))
                self.dfdu_form.append(diff(self.f_form, u))

                self.u_type.append('constant')

            else:

                self.dMdu_form.append(derivative(self.M_form, u, self.y))
                self.dfdu_form.append(derivative(self.f_form, u))

                self.u_type.append('function')

        def dMdu(t, x, y):

            self.t.value = t; self.x.vector()[:] = x; \
                              self.y.vector()[:] = y

            for expression in self.time_dependent_expresions:
                expression.t = t

            if self.u_type[0] == 'constant':

                A = numpy.zeros((self.dim, len(self.u_type)))

                for i in range(0, len(self.u_type)):

                    if self.u_type[i] == 'constant':
                        A[:, i] = self.assemble_M(self.dMdu_form[i]).dot(y)
                    else:
                        raise NameError('Mixing Constant and Function controls is not possible')

            else:
                if self.u_type[0] == 'function':

                    A = self.assemble_M(self.dMdu_form[0])

                    for i in range(1, len(self.u_type)):

                        if self.u_type[i] == 'function':
                            A = scipy.sparse.hstack([A, self.assemble_M(self.dMdu_form[i])])
                        else:
                            raise NameError('Mixing Constant and Function controls is not possible')

                else:
                    raise NameError('Control must be Constant or Function')

            return scipy.sparse.csc_matrix(A)

        def dfdu(t, x):

            self.t.value = t; self.x.vector()[:] = x


            for expression in self.time_dependent_expresions:
                expression.t = t

            if self.u_type[0] == 'constant':

                A = numpy.zeros((self.dim, len(self.u_type)))

                for i in range(0, len(self.u_type)):

                    if self.u_type[i] == 'constant':
                        A[:, i] = self.assemble_f(self.dfdu_form[i])
                    else:
                        raise NameError('Mixing Constant and Function controls is not possible')

            else:
                if self.u_type[0] == 'function':

                    A = self.assemble_M(self.dfdu_form[0])

                    for i in range(1, len(self.u_type)):

                        if self.u_type[i] == 'function':
                            A = scipy.sparse.hstack([A, self.assemble_M(self.dfdu_form[i])])
                        else:
                            raise NameError('Mixing Constant and Function controls is not possible')

                else:
                    raise NameError('Control must be Constant or Function')

            return scipy.sparse.csc_matrix(A)












        self.J_form = J_form
        self.g_form = g_form

        if self.J_form == None:
            J = None
        else:

            def J(t, x):

                self.t.value = t; self.x.vector()[:] = x

                for expression in self.time_dependent_expresions:
                    expression.t = t

                return assemble(self.J_form)

            self.dJdx_form = derivative(self.J_form, self.x)

            def dJdx(t, x):

                self.t.value = t; self.x.vector()[:] = x

                for expression in self.time_dependent_expresions:
                    expression.t = t

                return numpy.array(assemble(self.dJdx_form))

            self.dJdu_form = []

            for u in control:

                if type(u) == type(Constant(0.)):

                    self.dJdu_form.append(diff(self.J_form, u))
                else:

                    self.dJdu_form.append(derivative(self.J_form, u))

            def dJdu(t, x):

                self.t.value = t; self.x.vector()[:] = x


                for expression in self.time_dependent_expresions:
                    expression.t = t

                if self.u_type[0] == 'constant':

                    b = numpy.zeros(len(self.u_type))

                    for i in range(0, len(self.u_type)):

                        if self.u_type[i] == 'constant':
                            b[i] = assemble(self.dJdu_form[i])
                        else:
                            raise NameError('Mixing Constant and Function controls is not possible')

                else:
                    if self.u_type[0] == 'function':

                        b = numpy.array(assemble(self.dJdu_form[i]))

                        for i in range(1, len(self.u_type)):

                            if self.u_type[i] == 'function':
                                b = numpy.concatenate((b, numpy.array(assemble(self.dJdu_form[i]))))
                            else:
                                raise NameError('Mixing Constant and Function controls is not possible')

                    else:
                        raise NameError('Control must be Constant or Function')

                return b



        if self.g_form == None:
            g = None
        else:

            def g(t, x):

                self.t.value = t; self.x.vector()[:] = x

                for expression in self.time_dependent_expresions:
                    expression.t = t

                return assemble(self.g_form)

            self.dgdx_form = derivative(self.g_form, self.x)

            def dgdx(t, x):

                self.t.value = t; self.x.vector()[:] = x

                for expression in self.time_dependent_expresions:
                    expression.t = t

                return numpy.array(assemble(self.dgdx_form))

            self.dgdu_form = []

            for u in control:

                if type(u) == type(Constant(0.)):

                    self.dgdu_form.append(diff(self.g_form, u))
                else:

                    self.dgdu_form.append(derivative(self.g_form, u))

            def dgdu(t, x):

                self.t.value = t; self.x.vector()[:] = x


                for expression in self.time_dependent_expresions:
                    expression.t = t

                if self.u_type[0] == 'constant':

                    b = numpy.zeros(len(self.u_type))

                    for i in range(0, len(self.u_type)):

                        if self.u_type[i] == 'constant':
                            b[i] = assemble(self.dgdu_form[i])
                        else:
                            raise NameError('Mixing Constant and Function controls is not possible')

                else:
                    if self.u_type[0] == 'function':

                        b = numpy.array(assemble(self.dgdu_form[i]))

                        for i in range(1, len(self.u_type)):

                            if self.u_type[i] == 'function':
                                b = numpy.concatenate((b, numpy.array(assemble(self.dgdu_form[i]))))
                            else:
                                raise NameError('Mixing Constant and Function controls is not possible')

                    else:
                        raise NameError('Control must be Constant or Function')

                return b


        self.d2fdxdx_form = derivative(self.dfdx_form, self.x, self.y)

        def d2fdxdx(t, x, y):

            self.t.value = t; self.x.vector()[:] = x; self.y.vector()[:] = y

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.d2fdxdx_form)

        self.d2fdxdt_form = diff(self.dfdx_form, self.t_v)

        def d2fdxdt(t, x):

            self.t.value = t; self.x.vector()[:] = x

            for expression in self.time_dependent_expresions:
                expression.t = t

            return self.assemble_M(self.d2fdxdt_form)



        self.d2fdxdu_form = []

        for u in control:

            if type(u) == type(Constant(0.)):
                self.d2fdxdu_form.append(diff(self.dfdx_form, u))
            else:
                self.d2fdxdu_form.append(derivative(self.dfdx_form, u, self.y))

        def d2fdxdu(t, x, y):

            self.t.value = t; self.x.vector()[:] = x; \
                              self.y.vector()[:] = y

            for expression in self.time_dependent_expresions:
                expression.t = t

            if self.u_type[0] == 'constant':

                A = numpy.zeros((self.dim, len(self.u_type)))

                for i in range(0, len(self.u_type)):

                    if self.u_type[i] == 'constant':
                        A[:, i] = self.assemble_M(self.d2fdxdu_form[i]).dot(y)
                    else:
                        raise NameError('Mixing Constant and Function controls is not possible')

            else:
                if self.u_type[0] == 'function':

                    A = self.assemble_M(self.d2fdxdu_form[0])

                    for i in range(1, len(self.u_type)):

                        if self.u_type[i] == 'function':
                            A = scipy.sparse.hstack([A, self.assemble_M(self.d2fdxdu_form[i])])
                        else:
                            raise NameError('Mixing Constant and Function controls is not possible')

                else:
                    raise NameError('Control must be Constant or Function')

            return scipy.sparse.csc_matrix(A)

        self.d2fdtdu_form = []

        for u in control:

            if type(u) == type(Constant(0.)):
                self.d2fdtdu_form.append(diff(self.dfdt_form, u))
            else:
                self.d2fdtdu_form.append(derivative(self.dfdt_form, u))

        def d2fdtdu(t, x):

            self.t.value = t; self.x.vector()[:] = x


            for expression in self.time_dependent_expresions:
                expression.t = t

            if self.u_type[0] == 'constant':

                A = numpy.zeros((self.dim, len(self.u_type)))

                for i in range(0, len(self.u_type)):

                    if self.u_type[i] == 'constant':
                        A[:, i] = self.assemble_f(self.d2fdtdu_form[i])
                    else:
                        raise NameError('Mixing Constant and Function controls is not possible')

            else:
                if self.u_type[i] == 'vector':

                    A = self.assemble_M(self.d2fdtdu_form[0])

                    for i in range(1, len(self.u_type)):

                        if self.u_type[i] == 'function':
                            A = scipy.sparse.hstack([A, self.assemble_M(self.d2fdtdu_form[i])])
                        else:
                            raise NameError('Mixing Constant and Function controls is not possible')

                else:
                    raise NameError('Control must be Constant or Function')

            return scipy.sparse.csc_matrix(A)


        self.derivatives = {
                        'dMdx': self.dMdx,
                        'dMdt': self.dMdt,
                        'dMdu': dMdu,
                        'dfdx': self.dfdx,
                        'dfdt': self.dfdt,
                        'dfdu': dfdu,
                        'dJdx': dJdx,
                        'dJdu': dJdu,
                        'dgdx': dgdx,
                        'dgdu': dgdu,
                        'd2fdxdx': d2fdxdx,
                        'd2fdxdt': d2fdxdt,
                        'd2fdxdu': d2fdxdu,
                        'd2fdtdu': d2fdtdu
                        }

        fatDAE.class_problem.Control.__init__(self, self.M, self.f, x_0, t_0, t_f, J, g, self.derivatives)
