import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import fatdae.solvers
import fatdae.problem
import json
import scipy.sparse
import numpy
import scipy.optimize
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot

class DAE_SPM(fatdae.problem.Problem):

    def __init__(self, data, t_0, t_f):

        def M(t, x):
            A = numpy.array([[1., 0., 0., 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0., 0.], 
                             [0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0.]])

            return scipy.sparse.csc_matrix(A)
        
        self.L_n = data['negativeElectrode']['L']
        self.c_s_ini_n = data['negativeElectrode']['c_s_ini']
        self.c_s_max_n = data['negativeElectrode']['c_s_max']
        self.eps_e_n = data['negativeElectrode']['eps_e']
        self.R_p_n = data['negativeElectrode']['R_p']
        self.k_n = data['negativeElectrode']['k']
        self.U_RK_n = data['negativeElectrode']['U_RK']
        self.A_RK_n = data['negativeElectrode']['A_RK']
        self.D_s_n = lambda x: eval(data['negativeElectrode']['D_s'])

        self.L_p = data['positiveElectrode']['L']
        self.c_s_ini_p = data['positiveElectrode']['c_s_ini']
        self.c_s_max_p = data['positiveElectrode']['c_s_max']
        self.eps_e_p = data['positiveElectrode']['eps_e']
        self.R_p_p = data['positiveElectrode']['R_p']
        self.k_p = data['positiveElectrode']['k']
        self.U_RK_p = data['positiveElectrode']['U_RK']
        self.A_RK_p = data['positiveElectrode']['A_RK']
        self.D_s_p = lambda x: eval(data['positiveElectrode']['D_s'])

        self.c_e_ini = data['electrolyte']['c_e_ini']
        
        self.a_n = 3*self.eps_e_n/self.R_p_n
        self.a_p = 3*self.eps_e_p/self.R_p_p

        def f(t, x):
            R = 8.3144
            T = 298.15
            F = 96485.33

            j_n = - I_app(t)/(F*self.a_n*self.L_n)
            j_p = + I_app(t)/(F*self.a_p*self.L_p)

            i_n = self.k_n * self.c_e_ini ** 0.5 * (self.c_s_max_n - x[2]) ** 0.5 * x[2] ** 0.5
            i_p = self.k_p * self.c_e_ini ** 0.5 * (self.c_s_max_p - x[5]) ** 0.5 * x[5] ** 0.5

            eta_n = 2*(R*T/F)*numpy.arcsinh(j_n/i_n)
            eta_p = 2*(R*T/F)*numpy.arcsinh(j_p/i_p)

            x_sur_n = x[2]/self.c_s_max_n
            x_avg_n = x[0]/self.c_s_max_n

            x_sur_p = x[5]/self.c_s_max_p
            x_avg_p = x[3]/self.c_s_max_p

            b = numpy.array([- (3./self.R_p_n)*j_n, \
                             - 30.*(self.D_s_n(x_avg_n)/self.R_p_n**2)*x[1] - (45./2.)*(j_n/self.R_p_n**2), \
                             + 35.*(self.D_s_n(x_avg_n)/self.R_p_n)*(x[2]-x[0])-8.*self.D_s_n(x_avg_n)*x[1]+j_n, \
                             - (3./self.R_p_p)*j_p, \
                             - 30.*(self.D_s_p(x_avg_p)/self.R_p_p**2)*x[4] - (45./2.)*(j_p/self.R_p_p**2), \
                             + 35.*(self.D_s_p(x_avg_p)/self.R_p_p)*(x[5]-x[3])-8.*self.D_s_p(x_avg_p)*x[4]+j_p, \
                             x[6] - (self.U_p(x_sur_p) + eta_p - self.U_n(x_sur_n) - eta_n)
                            ])

            return b

        x_0 = numpy.array([self.c_s_ini_n, 0., self.c_s_ini_n, \
                           self.c_s_ini_p, 0., self.c_s_ini_p, \
                           self.U_p(self.c_s_ini_p/self.c_s_max_p) - self.U_n(self.c_s_ini_n/self.c_s_max_n)
                          ])

        super().__init__(M(t_0,x_0), f, x_0, t_0, t_f, derivatives={})

    def U_n(self,x):
        R = 8.3144
        T = 298.15
        F = 96485.33
        U = self.U_RK_n + (R * T / F) * numpy.log((1-x)/x)
        for k in range(len(self.A_RK_n)):
            U += (self.A_RK_n[k]/F)*((2*x-1)**(k+1)-(2*k*x*(1-x))/((2*x-1)**(1-k)))
        return U
    
    def U_p(self,x):
        R = 8.3144
        T = 298.15
        F = 96485.33
        U = self.U_RK_p + (R * T / F) * numpy.log((1-x)/x)
        for k in range(len(self.A_RK_p)):
            U += (self.A_RK_p[k]/F)*((2*x-1)**(k+1)-(2*k*x*(1-x))/((2*x-1)**(1-k)))
        return U
    
    def plot(self):
        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.t_list,numpy.array(self.x_list)[:,0],label='$c_{s,avg}^-$')
        matplotlib.pyplot.plot(self.t_list,numpy.array(self.x_list)[:,2],label='$c_{s,sur}^-$')
        matplotlib.pyplot.xlabel('Time [s]')
        matplotlib.pyplot.ylabel('Concentration [mol/m**3]')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.t_list,numpy.array(self.x_list)[:,3],label='$c_{s,avg}^+$')
        matplotlib.pyplot.plot(self.t_list,numpy.array(self.x_list)[:,5],label='$c_{s,sur}^+$')
        matplotlib.pyplot.xlabel('Time [s]')
        matplotlib.pyplot.ylabel('Concentration [mol/m**3]')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.t_list,numpy.array(self.x_list)[:,1],label='$q_{s,avg}^-$')
        matplotlib.pyplot.plot(self.t_list,numpy.array(self.x_list)[:,4],label='$q_{s,avg}^+$')
        matplotlib.pyplot.xlabel('Time [s]')
        matplotlib.pyplot.ylabel('Flux [mol/m**2 s]')
        matplotlib.pyplot.legend()

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.t_list,numpy.array(self.x_list)[:,6],'-*')
        matplotlib.pyplot.xlabel('Time [s]')
        matplotlib.pyplot.ylabel('Voltage [V]')

        matplotlib.pyplot.show()


with open(sys.path[0]+'/fatdae/json_butcher/RW/ROW4PW2.json') as data_file:
        butcher_json = json.load(data_file)

#with open(sys.path[0]+'/fatdae/json_butcher/SDIRK/SDIRK1.json') as data_file:
#        butcher_json = json.load(data_file)

embedded_1 = True
embedded_2 = False

solver = fatdae.solvers.build(butcher_json,embedded_1,embedded_2,a_tol=1e-12, r_tol=1e-2, h_min=1e-12,h_max=3e+2)

'''
For the negative electrode:
    c_{s,avg}^{-}'(t) + \frac{3}{R_p^-}j^-(t) = 0
    q_{s,avg}^{-}'(t) + 30\frac{D_s^-}{(R_p^-)^2}q_{s,avg}^{-}(t) + \frac{45}{2(R_p^-)^2}j^-(t) = 0
    35\frac{D_s^-}{R_p^-}(c_{s,sur}^-(t)-c_{s,avg}^{-}(t)) - 8D_s^-q_{s,avg}^{-}(t) + j^-(t) = 0
where j^-(t) = - I_{app}(t) / (F a_s^- L^-)

For the positive electrode:
    c_{s,avg}^{+}'(t) + \frac{3}{R_p^+}j^+(t) = 0
    q_{s,avg}^{+}'(t) + 30\frac{D_s^+}{(R_p^+)^2}q_{s,avg}^{+}(t) + \frac{45}{2(R_p^+)^2}j^+(t) = 0
    35\frac{D_s^+}{R_p^+}(c_{s,sur}^+(t)-c_{s,avg}^{+}(t)) - 8D_s^-q_{s,avg}^{+}(t) + j^+(t) = 0
where j^+(t) = + I_{app}(t) / (F a_s^+ L^+)

'''

with open('tests/testdata/spm_dae.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)

t_0 = 0
t_f = 2*3600

problem = DAE_SPM(data, t_0, t_f)

def I_app(t):
    t_cut = 3600
    k = 10
    I = 30
    return - I / (1+numpy.exp(-2*k*(t_cut-t)))

    if t > t_cut:
        return - I * numpy.exp(t_cut-t)
    else:
        return - I

problem.solve(solver, h=1, tlm=False, adp=True); problem.plot()