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
        scalar_element = FiniteElement('P', triangle, 1)
        vector_element = VectorElement('P', triangle, 2)
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
        return Expression('-x[0]+x[1]', degree=1)

    def exact_velocity(self):
        return Constant((0.0, 0.0))

    def define_functions(self):
        # Define functions for solutions at previous and current time steps
        u = Function(self.space)
        u = interpolate(self.ic, self.space)
        return u


class LidDrivenCavity():
    """
    """
    def __init__(self, nx, ny):
        self.mesh = self.make_mesh(nx, ny)
        self.space = self.make_space()
        self.bc = self.boundary_conditions()
        self.ic = self.initial_conditions()
        self.name = 'lid'

    def make_mesh(self, nx, ny): 
        return UnitSquareMesh(nx, ny)

    def make_space(self):
        scalar_element = FiniteElement('P', triangle, 1)
        vector_element = VectorElement('P', triangle, 2)
        mixed_element = MixedElement([scalar_element, vector_element])
        return FunctionSpace(self.mesh, MixedElement([scalar_element, vector_element]))

    def boundary_conditions(self):
        left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
        bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')

        v_wall = Constant((0, 0))
        v_lid = Constant((1, 0))
#        v_lid = Expression(('16.0*x[0]*x[0]*(1.0-x[0])*(1.0-x[0])', '0.0'), degree=4)

        v_bc1 = DirichletBC(self.space.sub(1), v_lid, top)
        v_bc2 = DirichletBC(self.space.sub(1), v_wall, bottom)
        v_bc3 = DirichletBC(self.space.sub(1), v_wall, left)
        v_bc4 = DirichletBC(self.space.sub(1), v_wall, right)
        return [v_bc1, v_bc2, v_bc3, v_bc4]

    def initial_conditions(self):
        return Constant((0, 0, 0))

    def define_functions(self):
        # Define functions for solutions at previous and current time steps
        u = Function(self.space)
        u = interpolate(self.ic, self.space)
        u0 = Function(self.space)
        u0.assign(u)
        return u, u0
