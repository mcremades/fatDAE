# Date: 23/06/2018
# Auth: Manuel Cremades, manuel.cremades@usc.es

# Basic modules
import sys; sys.path.insert(0,'..'); from base.basic_import import *

class Butcher(object):
    ''' Butcher table defining a Runge-Kutta method.

    .. inheritance-diagram:: Generalized
       :parts: 1

    The coefficients defining a Runge-Kutta method are given in form of a table:

    .. math::
        \\begin{equation}
            \\begin{array}{c|ccc}
            c_1     & a_{11}  & \\cdots & a_{1s}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            c_s     & a_{s1}  & \\cdots & a_{ss}  \\\\
            \\hline
                    & b_1     & \\cdots & b_s
            \\end{array}
            \\quad = \\quad
            \\begin{array}{c|c}
            \\mathbf{c} & A             \\\\
            \\hline
                        & \\mathbf{b}^T
            \\end{array}
        \\end{equation}

    were :math:`A\\in\\mathbb{R}^{s\\times s}`, :math:`\\mathbf{b}\\in\\mathbb{R}^{s}` and :math:`\\mathbf{c}\\in\\mathbb{R}^{s}`.

    .. note::
        If :math:`\\mathbf{c}\\in\\mathbb{R}^{s}` is not given, it will be computed from

        .. math::
            \\begin{equation}
                c_i = \\sum_{j=1}^{s}a_{ij}, \\quad i=1,\\dots, s
            \\end{equation}

        known as row condition.

    Attributes:
        A (:obj:`numpy.ndarray`): Matrix of coefficients, dimension :attr:`s` x :attr:`s`.
        b (:obj:`numpy.ndarray`): Vector of coefficients, dimension :attr:`s`.
        c (:obj:`numpy.ndarray`): Vector of coefficients, dimension :attr:`s`.
        p (:obj:`int`): Order of the method.
        s (:obj:`int`): Stages of the method.
    '''

    def __init__(self, json, embedded=False):
        ''' Instances the Butcher table.

        Args:
            json (:obj:`dict`): Dictionary defining the method.
        '''

        self.name = json["name"]

        self.A = numpy.array(eval(json["A"]))
        self.c = numpy.array(eval(json["c"]))

        if embedded == True:
            self.b = numpy.array(eval(json["b_2"]))
            self.p = json["p_2"]
        else:
            self.b = numpy.array(eval(json["b_1"]))
            self.p = json["p_1"]

        self.s = self.c.size

    def P(self, z):
        ''' Stability polynomial.

        it is defined as

        .. math::
            \\begin{equation}
                P(z) = \\det(I - zA + z(\\mathbb{1}\\otimes \\mathbf{b}))
            \\end{equation}

        were :math:`I\\in \\mathbb{R}^{s\\times s}` is an identity matrix and :math:`\\mathbb{1} = (1,\\dots, 1)^T\\in \\mathbb{R}^{s}`.

        Args:
            z (:obj:`complex`): Point to be evaluated.

        Returns:
            (:obj:`float`): Evaluation at the given point.
        '''

        return numpy.linalg.det(numpy.eye(self.s) - z * self.A + z * numpy.kron(numpy.ones(self.s).reshape(self.s, 1), self.b))

    def Q(self, z):
        ''' Stability polynomial.

        It is defined as

        .. math::
            \\begin{equation}
                Q(z) = \\det(I - zA)
            \\end{equation}

        were :math:`I\\in \\mathbb{R}^{s\\times s}` is an identity matrix.

        Args:
            z (:obj:`complex`): Point to be evaluated.

        Returns:
            (:obj:`float`): Evaluation at the given point.
        '''

        return numpy.linalg.det(numpy.eye(self.s) - z * self.A)

    def R(self, z):
        ''' Stability polynomial.

        It is defined as the quotient of two polynomials:

        .. math::
            \\begin{equation}
                R(z) = \\frac{P(z)}{Q(z)}
            \\end{equation}

        both defined in :meth:`P` and :meth:`Q` respectively.

        Args:
            z (:obj:`complex`): Point to be evaluated.

        Returns:
            (:obj:`float`): Evaluation at the given point.
        '''

        return self.P(z) / self.Q(z)

    def build_transposed(self):
        ''' Computes the table defining the transposed Runge-Kutta method.

        The transposed Runge-Kutta method is defined as

        .. math::
            \\begin{equation}
                a^t_{ij} = b_j\\frac{a_{ji}}{b_i}, \\quad b^t_i = b_i, \\quad c^t_i = 1 - c_i, \\quad i=1,\\dots, s
            \\end{equation}

        which it is usefull for adjoint computations.
        '''

        self.A_T = numpy.zeros((self.s, self.s))

        for i in range(self.s):
            for j in range(self.s):
                self.A_T[i, j] = self.b[j] * self.A[j, i] / self.b[i]

        self.c_T = numpy.zeros(self.s)

        for i in range(self.s):
            self.c_T[i] = 1.0 - self.c[i]

    def build_reflected(self):
        ''' Computes the table defining the reflected Runge-Kutta method.

        The reflected Runge-Kutta method is defined as

        .. math::
            \\begin{equation}
                a^r_{ij} = b_j - a_{ij}, \\quad b^r_i = b_i, \\quad c^r_i = 1 - c_i, \\quad i=1,\\dots, s
            \\end{equation}

        which it is usefull for adjoint computations.
        '''

        self.A_R = numpy.zeros((self.s, self.s))

        for i in range(self.s):
            for j in range(self.s):
                self.A_R[i, j] = self.b[j] - self.A[i, j]

        self.c_R = numpy.zeros(self.s)

        for i in range(self.s):
            self.c_R[i] = 1.0 - self.c[i]

class Parallel(Butcher):

    def __init__(self, json, embedded=False):
        ''' Instances the Butcher table.

        Args:
            json (:obj:`dict`): Dictionary defining the method.
        '''

        Butcher.__init__(self, json, embedded); self.D = numpy.array(eval(json["D"]))

class Generalized(Butcher):
    ''' Butcher table defining a Rosenbrock-Wanner method.

    The coefficients defining a Rosenbrock-Wanner method are given in form of a table:

    .. math::
        \\begin{equation}
            \\begin{array}{c|ccc|ccc|c}
            c_1     & a_{11}  & \\cdots & a_{1s}  & \\gamma_{11} & \\cdots & \\gamma_{1s} & d_1     \\\\
            \\vdots & \\vdots & \\ddots & \\vdots & \\vdots      & \\ddots & \\vdots      & \\vdots \\\\
            c_s     & a_{s1}  & \\cdots & a_{ss}  & \\gamma_{s1} & \\cdots & \\gamma_{ss} & d_s     \\\\
            \\hline
                    & b_1     & \\cdots & b_s     &              &         &              &
            \\end{array}
            \\quad = \\quad
            \\begin{array}{c|c|c|c}
            \\mathbf{c} & A             & G & \\mathbf{d} \\\\
            \\hline
                        & \\mathbf{b}^T &   &
            \\end{array}
        \\end{equation}

    were :math:`A \\in\\mathbb{R}^{s\\times s}`, :math:`G\\in\\mathbb{R}^{s\\times s}`, :math:`\\mathbf{b}\\in\\mathbb{R}^{s}`, :math:`\\mathbf{c}\\in\\mathbb{R}^{s}` and :math:`\\mathbf{d}\\in\\mathbb{R}^{s}`.

    .. note::
        If :math:`\\mathbf{c}\\in\\mathbb{R}^{s}` or :math:`\\mathbf{d}\\in\\mathbb{R}^{s}` are not given, they will be computed from

        .. math::
            \\begin{equation}
                c_i = \\sum_{j=1}^{s}a_{ij}, \\quad d_i = \\sum_{j=1}^{s}\gamma_{ij}, \\quad i=1,\\dots, s
            \\end{equation}

        known as row condition.

    Attributes:
        A (:obj:`numpy.ndarray`): Matrix of coefficients, dimension :attr:`s` x :attr:`s`.
        G (:obj:`numpy.ndarray`): Matrix of coefficients, dimension :attr:`s` x :attr:`s`.
        b (:obj:`numpy.ndarray`): Vector of coefficients, dimension :attr:`s`.
        c (:obj:`numpy.ndarray`): Vector of coefficients, dimension :attr:`s`.
        d (:obj:`numpy.ndarray`): Vector of coefficients, dimension :attr:`s`.
        p (:obj:`int`): Order of the method.
        s (:obj:`int`): Stage of the method.
    '''

    def __init__(self, json, embedded=False):
        ''' Instances the Butcher table.

        Args:
            json (:obj:`dict`): Dictionary defining the method.
        '''

        Butcher.__init__(self, json, embedded); self.G = numpy.array(eval(json["G"]))

        self.d = numpy.zeros(self.s)

        for i in range(self.s):
            self.d[i] = sum(self.G[i, :])

    def build_transposed(self):
        ''' Computes the table defining the transposed Rosenbrock-Wanner method.

        The transposed Runge-Kutta method is defined as

        .. math::
            \\begin{equation}
                a^t_{ij} = b_j\\frac{a_{ji}}{b_i}, \\quad b^t_i = b_i, \\quad c^t_i = 1 - c_i, \\quad i=1,\\dots, s
            \\end{equation}

        and

        .. math::
            \\begin{equation}
                \\gamma^t_{ij} = b_j\\frac{\\gamma_{ji}}{b_i}, \\quad i=1,\\dots, s
            \\end{equation}

        which it is usefull for adjoint computations.
        '''

        self.A_T = numpy.zeros((self.s, self.s))
        self.G_T = numpy.zeros((self.s, self.s))

        for i in range(self.s):
            for j in range(self.s):
                self.A_T[i, j] = self.b[j] * self.A[j, i] / self.b[i]
                self.G_T[i, j] = self.b[j] * self.G[j, i] / self.b[i]

        self.c_T = numpy.zeros(self.s)

        for i in range(self.s):
            self.c_T[i] = 1.0 - self.c[i]

if __name__ == '__main__':
    ''' Main.

    .. todo::
        Check order and plot stability.
    '''

    pass
