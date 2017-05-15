"""
Incompressible Navier-Stokes equations 

  rho(u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from fenics import *
from settings import reltol, maxiter

class Stokes():
    """
    """
    def __init__(self, _testcase, _alpha, _nu, u): 
        self.testcase = _testcase
        self.alpha = Constant(_alpha)
        self.nu = Constant(_nu)
        self.form, self.gain = self.variational_form(u)

    def variational_form(self, u):
        """
        Define the nonlinear variational problem to be solved
        """

        # Define trial and test functions
        test = TestFunction(self.testcase.space)
        du = TrialFunction(self.testcase.space)
        p, v = split(u)
        delp, delv = split(test)
        dp, dv = split(du)

        # Define expressions used in variational forms
        facets = FacetFunction('size_t', self.testcase.mesh)
        cells = CellFunction('size_t', self.testcase.mesh)

        da_ = Measure('ds', domain=self.testcase.mesh, subdomain_data=facets)
        dv_ = Measure('dx', domain=self.testcase.mesh, subdomain_data=cells)

        # Define form and gain for problem
        form = (self.alpha * dot(v, delv) + self.nu * dot(grad(v), grad(delv)) - p * div(delv) 
                - dot(self.testcase.f, delv) + delp * div(v)) * dv_

        gain = derivative(form, u, du)

        return form, gain

    def advance(self, u):
        # Advance the system to time t+dt by performing Newton-Raphson solve
        solve(self.form==0, u, self.testcase.bc, J=self.gain, \
                solver_parameters={"newton_solver":{"relative_tolerance":reltol, \
                        "maximum_iterations":maxiter}}, \
                form_compiler_parameters={"cpp_optimize":True, \
                        "representation":"quadrature", "quadrature_degree":2})
        return u
