"""
Incompressible Navier-Stokes equations 

  rho(u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from fenics import *
from settings import reltol, maxiter

def rate_of_strain(u):
    # Define strain-rate tensor
    return sym(nabla_grad(u))

def magnitude(tensor):
    # Define tensor magnitude
    return sqrt(inner(tensor, tensor))

class Abali():
    """
    """
    def __init__(self, _testcase, _dt, fluid, u, u0): 
        self.testcase = _testcase
        self.dt = Constant(_dt)
        self.form, self.gain = self.variational_form(fluid, u, u0)

    def variational_form(self, fluid, u, u0):
        """
        Define the nonlinear variational problem to be solved
        """

        # Define trial and test functions
        test = TestFunction(self.testcase.space)
        du = TrialFunction(self.testcase.space)
        p0, v0 = split(u0)
        p, v = split(u)
        delp, delv = split(test)
        dp, dv = split(du)

        # Define expressions used in variational forms
        facets = FacetFunction('size_t', self.testcase.mesh)
        cells = CellFunction('size_t', self.testcase.mesh)

        da_ = Measure('ds', domain=self.testcase.mesh, subdomain_data=facets)
        dv_ = Measure('dx', domain=self.testcase.mesh, subdomain_data=cells)

        II = 0.5 * inner(rate_of_strain(v), rate_of_strain(v)) + 1e-12

        # Define form and gain for problem
        form = (dot(div(v), delp) + fluid.rho / self.dt * dot((v - v0), delv)  \
                + fluid.rho * dot(dot(v, grad(v)), delv) + dot(grad(p), delv) \
                + fluid.apparent_viscosity(II) * inner(rate_of_strain(v), grad(delv))) * dv_

        gain = (dot(div(dv), delp) + fluid.rho / self.dt * dot(dv, delv) \
                + fluid.rho * dot(dot(dv, grad(v)), delv) \
                + fluid.rho * dot(dot(v, grad(dv)), delv) \
                + dot(grad(dp), delv) \
                + fluid.apparent_viscosity(II) * inner(rate_of_strain(dv), grad(delv)) \
                + fluid.differentiated_apparent_viscosity(II) \
                * inner(rate_of_strain(v), rate_of_strain(dv)) \
                * inner(rate_of_strain(v), grad(delv))) * dv_

        return form, gain

    def advance(self, u):
        # Advance the system to time t+dt by performing Newton-Raphson solve
        solve(self.form==0, u, self.testcase.bc, J=self.gain, \
                solver_parameters={"newton_solver":{"relative_tolerance":reltol, \
                        "maximum_iterations":maxiter}}, \
                form_compiler_parameters={"cpp_optimize":True, \
                        "representation":"quadrature", "quadrature_degree":2})
        return u
