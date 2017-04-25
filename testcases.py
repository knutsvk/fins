from fenics import * 


class Poiseuille():
    """
    """
    def __init__(self, nx=20, ny=20, xmin=0, xmax=1, ymin=0, ymax=1, dp=1):
        self.mesh = self.make_mesh(nx, ny)
        self.space = self.make_space()
        self.bc = self.boundary_conditions()
        self.ic = self.initial_conditions()
        self.name = 'poiseuille'

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

        v_noslip = Constant((0, 0))
        pL = Constant('1.0')
        pR = Constant('0.0')

        p_bc1 = DirichletBC(self.space.sub(0), pL, left)
        p_bc2 = DirichletBC(self.space.sub(0), pR, right)

        v_bc1 = DirichletBC(self.space.sub(1), v_noslip, bottom)
        v_bc2 = DirichletBC(self.space.sub(1), v_noslip, top)
        v_bc3 = DirichletBC(self.space.sub(1).sub(1), 0, left)
        v_bc4 = DirichletBC(self.space.sub(1).sub(1), 0, right)
        return [p_bc1, p_bc2, v_bc1, v_bc2, v_bc3, v_bc4]

    def initial_conditions(self):
        return Expression(('1.0-x[0]', '0.0', '0.0'), degree=1)

    def define_functions(self):
        # Define functions for solutions at previous and current time steps
        u = Function(self.space)
        u = interpolate(self.ic, self.space)
        u0 = Function(self.space)
        u0.assign(u)
        return u, u0
