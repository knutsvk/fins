"""
Incompressible Navier-Stokes equations

  rho(u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from fenics import *

class GSP():
    """
    Generalised Stokes Problem
    """
    def __init__(self, _testcase, _alpha, _nu, U):
        self.testcase = _testcase
        self.alpha = Constant(_alpha)
        self.nu = Constant(_nu)
        self.form, self.gain = self.variational_form(U)

    def variational_form(self, U):
        """
        Define the nonlinear variational problem to be solved
        """

        # Define trial and test functions
        test = TestFunction(self.testcase.space)
        q, v = split(test)
        dU = TrialFunction(self.testcase.space)
        p, u = split(U)

        # Define expressions used in variational forms
        facets = FacetFunction('size_t', self.testcase.mesh)
        cells = CellFunction('size_t', self.testcase.mesh)

        da_ = Measure('ds', domain=self.testcase.mesh, subdomain_data=facets)
        dv_ = Measure('dx', domain=self.testcase.mesh, subdomain_data=cells)

        # Define form and gain for problem
        form = (self.alpha * dot(u, v)
                + self.nu * inner(grad(u), grad(v))
                - p * div(v)
                - dot(self.testcase.f, v)
                + q * div(u)) * dv_

        gain = derivative(form, U, dU)

        return form, gain

    def solve(self, U):
        # Advance the system to time t+dt by performing Newton-Raphson solve
        p, u = U.split(deepcopy=True)
        goal = div(u)*dx()
        tol = 1e-12

        problem = NonlinearVariationalProblem(self.form, U, self.testcase.bc, J=self.gain,
                form_compiler_parameters={"optimize": True, "cpp_optimize": True,
                    "representation": "quadrature", "quadrature_degree": 2})
        solver = AdaptiveNonlinearVariationalSolver(problem, goal)
#        solver.parameters["newton_solver"]["linear_solver"] = "cg"
#        solver.parameters["newton_solver"]["preconditioner"] = "icc"
        solver.solve(tol)

        return k
