"""
Generalized Stokes Problem

  alpha u + nu lap(u) - grad(p) = f
                         div(u) = 0
"""

from fenics import *

class GSP():
    """
    Generalised Stokes Problem
    """
    def __init__(self, _testcase, _alpha, _nu):
        self.testcase = _testcase
        self.alpha = Constant(_alpha)
        self.nu = Constant(_nu)
        self.a, self.L = self.variational_form(self.testcase.f)

    def variational_form(self, f):
        """
        Define the linear variational problem to be solved
        """

        # Define trial and test functions
        q, v = TestFunctions(self.testcase.space)
        p, u = TrialFunctions(self.testcase.space)

        # Define bilinear and linear forms (resp. a and L)
        a = (self.alpha * inner(u, v)
                + self.nu * inner(grad(u), grad(v))
                - p * div(v)
                + q * div(u)) * dx()

        L = inner(f, v) * dx()

        return a, L

    def solve(self, U):
        """
        """
        problem = LinearVariationalProblem(self.a, self.L, U, self.testcase.bc)
        solver = LinearVariationalSolver(problem)
        solver.solve()
