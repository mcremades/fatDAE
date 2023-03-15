from setuptools import setup

setup(
    name='fatDAE',
    version='0.1.0',    
    description='Runge-Kutta and Rosenbrock-Wanner methods for solving linearly (or quasi-linearly) implicit problems of the form M(t,y)dy/dt(t)=f(t,y) with adjoint state and tangent linear model features.',
    url='https://github.com/mcremades/fatDAE',
    author='Manuel Cremades',
    author_email='manuel.cremades@usc.es',
    license='GNU Lesser General Public License v3.0',
    packages=['fatDAE'],
    install_requires=['numpy',  
                      'scipy',
                      'sympy',
                      'matplotlib',
                      'argparse',
                      'openpyxl'                   
                      ],
)
