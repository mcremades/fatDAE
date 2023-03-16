import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import fatdae.solvers
import fatdae.problem
import json
import scipy.sparse
import numpy
import scipy.optimize

with open(sys.path[0]+'/fatdae/json_butcher/RW/ROW4PW2.json') as data_file:
        butcher_json = json.load(data_file)

#with open(sys.path[0]+'/fatdae/json_butcher/SDIRK/SDIRK1.json') as data_file:
#        butcher_json = json.load(data_file)

embedded_1 = True
embedded_2 = False

solver = fatdae.solvers.build(butcher_json,embedded_1,embedded_2,a_tol=1e-12, r_tol=1e-3, h_min=1e-12,h_max=1e+2)

'''
Single particle model 

For the negative electrode:
    c_{s,avg}^{-}'(t) = -\frac{3}{R_s^-}j^-(t)
    5\frac{D_s^-}{R_s^-}(c_{s,sur}^-(t)-c_{s,avg}^{-}(t)) = -j^-(t)
where j^-(t) = - I_{app}(t) /

For the positive electrode:
    c_{s,avg}^{-}'(t) = -\frac{3}{R_s^-}j^+(t)
    5\frac{D_s^-}{R_s^-}(c_{s,sur}^-(t)-c_{s,avg}^{-}(t)) = -j^+(t)
where j^+(t) = + I_{app}(t) / 


'''

F = 96485.33 
R = 8.34
T = 298.15

L_n = 1.28e-4
eps_e_n = 0.434

R_n = 0.5e-5
D_n = lambda x: 3.9e-14
c_n_ini = 2.1473e4
c_n_max = 2.6390e4

U_n_0 = -1.7203
A_n = [-0.35799e6,-0.35008e6,-0.35247e6,-0.35692e6,-0.38633e6,-0.35908e6,-0.28794e6,-0.14979e6,-0.39912e6,-0.96172e6,-0.63262e6]
def U_n(x):
    U = U_n_0 + (R * T / F) * numpy.log((1-x)/x)
    for k in range(len(A_n)):
        U += (A_n[k]/F)*((2*x-1)**(k+1)-(2*k*x*(1-x))/((2*x-1)**(1-k)))
    return float(U)
k_n = 1e-6



L_p =  9.1e-5
eps_e_p = 0.400

R_p = 2.5e-6
D_p = lambda x: 3e-15*(1+numpy.tanh(-20*x+14.6)+0.02)
c_p_ini = 6.5632e3 
c_p_max = 4.9459e4
U_p_0 = 3.966,
A_p = [-5.8992e4,-2.0881e4,-1.3273e4,-6.9538e3,-2.6023e4,1.0715e4]
def U_p(x):
    U = U_p_0 + (R * T / F) * numpy.log((1-x)/x)
    for k in range(len(A_p)):
        U += (A_p[k]/F)*((2*x-1)**(k+1)-(2*k*x*(1-x))/((2*x-1)**(1-k)))
    return float(U)
k_p = 1e-6

#R_p = 1.7e-6
#D_p = lambda x: 6.0e-15
#c_p_ini = 3.9509e3
#c_p_max = 2.3887e4
#U_p = lambda x: 4.0 - x


c_e_ini = 1000


def I_app(t):
    t_cut = 1800
    k = 10
    I = 50
    return - I / (1+numpy.exp(-2*k*(t_cut-t)))

    if t > t_cut:
        return - I * numpy.exp(t_cut-t)
    else:
        return - I

def M(t, x):

    A = numpy.array([[1., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0.], 
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 0., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 0.]])

    return scipy.sparse.csc_matrix(A)

def f(t, x):

    a_n = 3*eps_e_n/R_n
    a_p = 3*eps_e_p/R_p

    j_n = - I_app(t)/(F*a_n*L_n)
    j_p = + I_app(t)/(F*a_p*L_p)

    i_n = k_n * c_e_ini ** 0.5 * (c_n_max - x[2]) ** 0.5 * x[2] ** 0.5
    i_p = k_p * c_e_ini ** 0.5 * (c_p_max - x[5]) ** 0.5 * x[5] ** 0.5

    eta_n = 2*(R*T/F)*numpy.arcsinh(j_n/i_n)
    eta_p = 2*(R*T/F)*numpy.arcsinh(j_p/i_p)
    
    b = numpy.array([- (3./R_n)*j_n, \
                     - 30.*(D_n(x[0]/c_n_max)/R_n**2)*x[1] - (45./2.)*(j_n/R_n**2), \
                     35.*(D_n(x[0]/c_n_max)/R_n)*(x[2]-x[0])-8.*D_n(x[0]/c_n_max)*x[1]+j_n, \
                     - (3/R_p)*j_p, \
                    - 30.*(D_p(x[3]/c_p_max)/R_n**2)*x[4] - (45./2.)*(j_p/R_p**2), \
                     35.*(D_p(x[3]/c_p_max)/R_p)*(x[5]-x[3])-8.*D_p(x[3]/c_p_max)*x[4]+j_p, \
                     x[6] - (U_p(x[5]/c_p_max) + eta_p - U_n(x[2]/c_n_max) - eta_n)
                    ]
                    )

    return b

#def dfdx(t, x):
#
#    A = numpy.array([[0., 0., 0., 0.], \
#                     [-5*(D_n/R_n), 5*(D_n/R_n), 0., 0.],
#                     [0., 0., 0., 0.],
#                     [0., 0., -5*(D_p/R_p), 5*(D_p/R_p)]]
#                    )
#
#    return scipy.sparse.csc_matrix(A)

#def dfdt(t, x):
#
#    b = numpy.array([0., \
#                     0., \
#                     0., \
#                     0.]
#                    )
#
#    return b

x_0 = numpy.array([c_n_ini, \
                   0.,\
                   c_n_ini, \
                   c_p_ini, \
                   0.,\
                   c_p_ini, \
                   U_p(c_p_ini/c_p_max) - U_n(c_n_ini/c_n_max)
                  ]
                  )


#derivatives = {'dfdx': dfdx, \
#               'dfdt': dfdt}

derivatives = {}

problem = fatdae.problem.Problem(M(0., x_0), f, x_0, 0., 3600., derivatives)

problem.solve(solver, h=1, tlm=False, adp=True); problem.plot()