from abc import ABC, abstractmethod
from fenics import * 
import numpy as np


class Problem(ABC):
    """
    Abstract class which describes the specific problem for which the Stokes flow equations  are
    solved. The specific problem must have a name and define a mesh, function space(s), boundary
    conditions and external forces. Optionally, analytical solutions and error computers may be
    specified. 
    """
    def __init__(self, nx, ny, _name):
        self.mesh = self.make_mesh(nx, ny)
        self.space = self.make_space()
        self.bc = self.boundary_conditions()
        self.f = self.external_forces()
        self.name = _name
    
    @staticmethod
    @abstractmethod
    def make_mesh(nx, ny):
        return UnitSquareMesh(nx, ny)

    @abstractmethod
    def make_space(self):
        scalar_element = FiniteElement('Lagrange', triangle, 1)
        vector_element = VectorElement('Lagrange', triangle, 2)
        mixed_element = MixedElement([scalar_element, vector_element])
        return FunctionSpace(self.mesh, mixed_element)

    @abstractmethod
    def boundary_conditions(self):
        pass

    @staticmethod
    @abstractmethod
    def external_forces():
        return Constant((0, 0))

    def define_functions(self):
        u = Function(self.space)
        return u

    def write_file(self, U, nx, ny, outdir='./output/'):
        p, u = U.split(deepcopy=True)
        file_p = File(outdir + self.name + '_nx' + str(nx) + '_ny' + str(ny) + '_' +
                'pressure.pvd')
        file_u = File(outdir + self.name + '_nx' + str(nx) + '_ny' + str(ny) + '_' +
                'velocity.pvd')
        file_p << p
        file_u << u


class Smooth(Problem):
    """
    Smooth test case as described on page 884 in: 
    J. Cahouet and J.-P. Chabard (1988)
    "Some fast 3D finite element solvers for the generalized Stokes problem"
    International Journal for Numerical Methods in Fluids, vol. 8, 869-895
    """
    def __init__(self, nx, ny):
        super().__init__(nx, ny, 'smooth')

    def make_mesh(self, nx, ny):
        return super().make_mesh(nx, ny)

    def make_space(self):
        return super().make_space()

    def boundary_conditions(self):
        left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
        bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')
        bcs = [left, right, bottom, top]
        return [DirichletBC(self.space.sub(1), Constant((0, 0)), bc) for bc in bcs]

    @staticmethod
    def external_forces():
        return Constant((-1, 1))

    @staticmethod
    def exact_pressure():
        return Expression('-x[0]+x[1]', degree=1)

    @staticmethod
    def exact_velocity():
        return Constant((0, 0))

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


class PolkaDots(Problem):
    """
    Analytical test case as described on page 390 in: 
    A. Logg et al. (2012)
    "Automated Solution of Differential Equations by the Finite Element Method"
    Springer-Verlag Berlin Heidelberg
    """
    def __init__(self, nx, ny):
        super().__init__(nx, ny, 'polka')

    def make_mesh(self, nx, ny): 
        return UnitSquareMesh(nx, ny, 'crossed')

    def make_space(self):
        return super().make_space()

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

    @staticmethod
    def external_forces():
        return Expression(('28*pi*pi * sin(4*pi*x[0]) * cos(4*pi*x[1])', 
            '- 36*pi*pi * cos(4*pi*x[0]) * sin(4*pi*x[1])'), degree=10)

    @staticmethod
    def exact_pressure():
        return Expression('pi * cos(4*pi*x[0]) * cos(4*pi*x[1])', degree=10)

    @staticmethod
    def exact_velocity():
        return Expression(('sin(4*pi*x[0]) * cos(4*pi*x[1])', 
            '- cos(4*pi*x[0]) * sin(4*pi*x[1])'), degree=10)

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


class LidDrivenCavity(Problem):
    """
    Analytical test case as described on page 390 in: 
    A. Logg et al. (2012)
    "Automated Solution of Differential Equations by the Finite Element Method"
    Springer-Verlag Berlin Heidelberg
    """
    def __init__(self, nx, ny):
        super().__init__(nx, ny, 'lid')

    def make_mesh(self, nx, ny): 
        return super().make_mesh(nx, ny)

    def make_space(self):
        return super().make_space()

    def boundary_conditions(self):
        pin = CompiledSubDomain('near(x[0], 0.0)  && on_boundary')
        p_bc = DirichletBC(self.space.sub(0), Constant(0), pin)

        left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
        bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')
        bcs = [left, right, bottom, top]

        u_wall = Constant((0, 0))
        u_lid = Expression(('x[0]*(1.0-x[0])', '0.0'), degree=2)
        vels = [u_wall, u_wall, u_wall, u_lid]

        return [DirichletBC(self.space.sub(1), vel, bc) for (vel, bc) in zip(vels, bcs)]

    def external_forces(self):
        return super().external_forces()


class Sinflow3D():
    """
    3D test case with a sinusoidal inflow as described in: 
    https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/stokes-iterative/python/documentation.html
    """
    def __init__(self, nx, ny, nz):
        self.mesh = self.make_mesh(nx, ny, nz)
        self.space = self.make_space()
        self.bc = self.boundary_conditions()
        self.f = self.external_forces()
        self.name = 'sinflow'

    def make_mesh(self, nx, ny, nz): 
        return UnitCubeMesh(nx, ny, nz)

    def make_space(self):
        scalar_element = FiniteElement('Lagrange', tetrahedron, 1)
        vector_element = VectorElement('Lagrange', tetrahedron, 2)
        mixed_element = MixedElement([scalar_element, vector_element])
        return FunctionSpace(self.mesh, mixed_element)

    def boundary_conditions(self):
        # Boundaries
        def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
        def left(x, on_boundary): return x[0] < DOLFIN_EPS
        def top_bottom(x, on_boundary): return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

        # No-slip boundary condition for velocity
        noslip = Constant((0.0, 0.0, 0.0))
        bc0 = DirichletBC(self.space.sub(1), noslip, top_bottom)

        # Inflow boundary condition for velocity
        inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=10)
        bc1 = DirichletBC(self.space.sub(1), inflow, right)

        # Boundary condition for pressure at outflow
        bc2 = DirichletBC(self.space.sub(0), Constant(0), left)

        # Collect boundary conditions
        return [bc0, bc1, bc2]

    def write_file(self, U, n, outdir='./output/'):
        p, u = U.split(deepcopy=True)
        file_p = File(outdir + self.name + '_n' + str(n) + '_' + 'pressure.pvd')
        file_u = File(outdir + self.name + '_n' + str(n) + '_' + 'velocity.pvd')
        file_p << p
        file_u << u

    def external_forces(self):
        return Constant((0, 0, 0))

    def define_functions(self):
        # Define functions for solutions at previous and current time steps
        U = Function(self.space)
        return U
