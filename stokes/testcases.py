from fenics import * 

class Smooth():
    """
    """
    def __init__(self, nx, ny, limits=[0, 1, 0, 1], dp=1):
        self.mesh = self.make_mesh(nx, ny, limits)
        self.space = self.make_space()
        self.bc = self.boundary_conditions(limits, dp)
        self.ic = self.initial_conditions(limits, dp)
        self.force = self.external_forces(limits, dp)
        self.name = 'smooth'

    def make_mesh(self, nx, ny, limits): 
        return UnitSquareMesh(nx, ny)

    def make_space(self):
        scalar_element = FiniteElement('P', triangle, 1)
        vector_element = VectorElement('P', triangle, 2)
        mixed_element = MixedElement([scalar_element, vector_element])
        return FunctionSpace(self.mesh, MixedElement([scalar_element, vector_element]))

    def boundary_conditions(self, limits, dp):
        left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
        right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
        bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
        top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')

        v_noslip = Constant((0, 0))

        v_bc1 = DirichletBC(self.space.sub(1), v_noslip, bottom)
        v_bc2 = DirichletBC(self.space.sub(1), v_noslip, left)
        v_bc3 = DirichletBC(self.space.sub(1), v_noslip, top)
        v_bc4 = DirichletBC(self.space.sub(1), v_noslip, right)
        return [v_bc1, v_bc2, v_bc3, v_bc4]

    def initial_conditions(self, limits, dp):
        return Constant((0.0, 0.0, 0.0))

    def external_forces(self, limits, dp):
        return Constant((-1.0, 1.0))

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
