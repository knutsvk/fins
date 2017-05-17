from fenics import * 
import numpy as np

class Smooth():
    """
    Smooth test case as described on page 884 in: 
    J. Cahouet and J.-P. Chabard (1988)
    "Some fast 3D finite element solvers for the generalized Stokes problem"
    International Journal for Numerical Methods in Fluids, vol. 8, 869-895
    """
    def __init__(self, nx, ny):
        self.mesh = self.make_mesh(nx, ny)
        self.space = self.make_space()
        self.bc = self.boundary_conditions()
        self.ic = self.initial_conditions()
        self.f = self.external_forces()
        self.name = 'smooth'

    def make_mesh(self, nx, ny): 
        return UnitSquareMesh(nx, ny)

    def make_space(self):
        scalar_element = FiniteElement('Lagrange', triangle, 1)
        vector_element = VectorElement('Lagrange', triangle, 2)
        mixed_element = MixedElement([scalar_element, vector_element])
        return FunctionSpace(self.mesh, MixedElement([scalar_element, vector_element]))

    def boundary_conditions(self):
        left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
        bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')
        bcs = [left, right, bottom, top]
        return [DirichletBC(self.space.sub(1), Constant((0, 0)), bc) for bc in bcs]

    def initial_conditions(self):
        return Constant((0.0, 0.0, 0.0))

    def external_forces(self):
        return Constant((-1.0, 1.0))

    def compute_errors(self, U):
        p, u = U.split(deepcopy=True)

        vertex_values_p = p.compute_vertex_values(self.mesh)
        vertex_values_pe = self.exact_pressure().compute_vertex_values(self.mesh)
        diff_p = vertex_values_p - np.mean(vertex_values_p) - vertex_values_pe

        vertex_values_u = u.compute_vertex_values(self.mesh)
        vertex_values_ue = self.exact_velocity().compute_vertex_values(self.mesh)
        diff_u = vertex_values_u - vertex_values_ue

        norms = [1, 2, np.inf]
        error_norms_p = [np.linalg.norm(diff_p, norm) for norm in norms]
        error_norms_u = [np.linalg.norm(diff_u, norm) for norm in norms]

        return error_norms_p, error_norms_u

    def exact_pressure(self):
        """
        Analytical solution for the pressure in this testcase
        """
        return Expression('-x[0]+x[1]', degree=1)

    def exact_velocity(self):
        """
        Analytical solution for the velocity in this testcase
        """
        return Constant((0.0, 0.0))

    def define_functions(self):
        """
        Define function to hold solution 
        """
        u = Function(self.space)
        u = interpolate(self.ic, self.space)
        return u


class PolkaDots():
    """
    Analytical test case as described on page 390 in: 
    A. Logg et al. (2012)
    "Automated Solution of Differential Equations by the Finite Element Method"
    Springer-Verlag Berlin Heidelberg
    """
    def __init__(self, nx, ny):
        self.mesh = self.make_mesh(nx, ny)
        self.space = self.make_space()
        self.bc = self.boundary_conditions()
        self.ic = self.initial_conditions()
        self.f = self.external_forces()
        self.name = 'polka'

    def make_mesh(self, nx, ny): 
        return UnitSquareMesh(nx, ny, 'crossed')

    def make_space(self):
        scalar_element = FiniteElement('Lagrange', triangle, 1)
        vector_element = VectorElement('Lagrange', triangle, 2)
        mixed_element = MixedElement([scalar_element, vector_element])
        return FunctionSpace(self.mesh, MixedElement([scalar_element, vector_element]))

    def boundary_conditions(self):
        pin = CompiledSubDomain('near(x[0], 0.0)  && on_boundary')
        p_bc = DirichletBC(self.space.sub(0), self.exact_pressure(), pin)

        left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
        bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')

        u_bc1 = DirichletBC(self.space.sub(1), self.exact_velocity(), left)
        u_bc2 = DirichletBC(self.space.sub(1), self.exact_velocity(), right)
        u_bc3 = DirichletBC(self.space.sub(1), self.exact_velocity(), bottom)
        u_bc4 = DirichletBC(self.space.sub(1), self.exact_velocity(), top)

        return [p_bc, u_bc1, u_bc2, u_bc3, u_bc4]

    def initial_conditions(self):
        return Constant((0, 0, 0))

    def external_forces(self):
        return Expression(('28*pi*pi * sin(4*pi*x[0]) * cos(4*pi*x[1])', 
            '- 36*pi*pi * cos(4*pi*x[0]) * sin(4*pi*x[1])'), degree=10)

    def compute_errors(self, U):
        p, u = U.split(deepcopy=True)

        vertex_values_p = p.compute_vertex_values(self.mesh)
        vertex_values_pe = self.exact_pressure().compute_vertex_values(self.mesh)
        diff_p = vertex_values_p - vertex_values_pe

        vertex_values_u = u.compute_vertex_values(self.mesh)
        vertex_values_ue = self.exact_velocity().compute_vertex_values(self.mesh)
        diff_u = vertex_values_u - vertex_values_ue

        norms = [2, np.inf]
        error_norms_p = [np.linalg.norm(diff_p, norm) for norm in norms]
        error_norms_u = [np.linalg.norm(diff_u, norm) for norm in norms]

        return error_norms_p, error_norms_u

    def exact_pressure(self):
        """
        Analytical solution for the pressure in this testcase
        """
        return Expression('pi * cos(4*pi*x[0]) * cos(4*pi*x[1])', degree=10)

    def exact_velocity(self):
        """
        Analytical solution for the velocity in this testcase
        """
        return Expression(('sin(4*pi*x[0]) * cos(4*pi*x[1])', 
            '- cos(4*pi*x[0]) * sin(4*pi*x[1])'), degree=10)

    def define_functions(self):
        """
        Define function to hold solution 
        """
        U = Function(self.space)
        U = interpolate(self.ic, self.space)
        return U


class LidDrivenCavity():
    """
    Analytical test case as described on page 390 in: 
    A. Logg et al. (2012)
    "Automated Solution of Differential Equations by the Finite Element Method"
    Springer-Verlag Berlin Heidelberg
    """
    def __init__(self, nx, ny):
        self.mesh = self.make_mesh(nx, ny)
        self.space = self.make_space()
        self.bc = self.boundary_conditions()
        self.ic = self.initial_conditions()
        self.f = self.external_forces()
        self.name = 'lid'

    def make_mesh(self, nx, ny): 
        return UnitSquareMesh(nx, ny)

    def make_space(self):
        scalar_element = FiniteElement('P', triangle, 1)
        vector_element = VectorElement('P', triangle, 2)
        mixed_element = MixedElement([scalar_element, vector_element])
        return FunctionSpace(self.mesh, MixedElement([scalar_element, vector_element]))

    def boundary_conditions(self):
        pin = CompiledSubDomain('near(x[0], 0.0)  && on_boundary')
        p_bc = DirichletBC(self.space.sub(0), Constant(0), pin)

        left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
        bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')

        u_wall = Constant((0, 0))
        u_lid = Expression(('x[0]*(1.0-x[0])', '0.0'), degree=2)

        u_bc1 = DirichletBC(self.space.sub(1), u_lid, top)
        u_bc2 = DirichletBC(self.space.sub(1), u_wall, bottom)
        u_bc3 = DirichletBC(self.space.sub(1), u_wall, left)
        u_bc4 = DirichletBC(self.space.sub(1), u_wall, right)
        return [u_bc1, u_bc2, u_bc3, u_bc4]

    def initial_conditions(self):
        return Constant((0, 0, 0))

    def external_forces(self):
        return Constant((-1.0, 1.0))

    def define_functions(self):
        # Define functions for solutions at previous and current time steps
        U = Function(self.space)
        U = interpolate(self.ic, self.space)
        return U
