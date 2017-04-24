from fenics import *
from rheology import Newtonian, Sigmoid, Papabing
set_log_level(ERROR)

nx = ny = 32
mesh = UnitSquareMesh(nx, ny)

scalar_element = FiniteElement('P', triangle, 1)
vector_element = VectorElement('P', triangle, 2)
mixed_element = MixedElement([scalar_element, vector_element])
Space = FunctionSpace(mesh, mixed_element)

left = CompiledSubDomain('near(x[0], 0.0) && on_boundary')
right = CompiledSubDomain('near(x[0], 1.0) && on_boundary')
bottom = CompiledSubDomain('near(x[1], 0.0) && on_boundary')
top = CompiledSubDomain('near(x[1], 1.0) && on_boundary')

facets = FacetFunction('size_t', mesh)
cells = CellFunction('size_t', mesh)
da_ = Measure('ds', domain=mesh, subdomain_data=facets)
dv_ = Measure('dx', domain=mesh, subdomain_data=cells)

v_noslip = Constant((0, 0))
pL = Constant('1.0')
pR = Constant('0.0')

p_bc1 = DirichletBC(Space.sub(0), pL, left)
p_bc2 = DirichletBC(Space.sub(0), pR, right)
v_bc1 = DirichletBC(Space.sub(1), v_noslip, bottom)
v_bc2 = DirichletBC(Space.sub(1), v_noslip, top)
v_bc3 = DirichletBC(Space.sub(1).sub(1), 0, left)
v_bc4 = DirichletBC(Space.sub(1).sub(1), 0, right)

bc = [p_bc1, p_bc2, v_bc1, v_bc2, v_bc3, v_bc4]
u_init = Expression(('1.0-x[0]', '0.0', '0.0'), degree=1)

t = 0.0
t_end = 1.0
dt = 0.05

test = TestFunction(Space)
du = TrialFunction(Space)
u0 = Function(Space)
u = Function(Space)
u = interpolate(u_init, Space)
u0.assign(u)
p0, v0 = split(u0)
p, v = split(u)
delp, delv = split(test)
dp, dv = split(du)

rho = 1.0
mu = 1.0
tau_y = 0.3
eps = 1e-4

fluid = Newtonian(rho, mu)
fluid = Papabing(rho, mu, tau_y, eps)
fluid = Sigmoid(rho, mu, tau_y, eps)

def rate_of_strain(u):
    return sym(nabla_grad(u))

II = 0.5 * inner(rate_of_strain(v), rate_of_strain(v)) + 1e-12

Form = (dot(div(v), delp) + rho / dt * dot((v - v0), delv)  \
        + rho * dot(dot(v, grad(v)), delv) + dot(grad(p), delv) \
        + fluid.apparent_viscosity(II) * inner(rate_of_strain(v), grad(delv))) * dv_

Gain = (dot(div(dv), delp) + rho / dt * dot(dv, delv) \
        + rho * dot(dot(dv, grad(v)), delv) \
        + rho * dot(dot(v, grad(dv)), delv) \
        + dot(grad(dp), delv) \
        + fluid.apparent_viscosity(II) * inner(rate_of_strain(dv), grad(delv)) \
        + fluid.differentiated_apparent_viscosity(II) \
        * inner(rate_of_strain(v), rate_of_strain(dv)) \
        * inner(rate_of_strain(v), grad(delv))) * dv_

pwd = './viscoplastic/'
file_p = File(pwd + 'pressure.pvd')
file_v = File(pwd + 'velocity.pvd')

step = 0
print('step\t time\t umax\t change')
while t < t_end:
    step += 1
    t += dt
    solve(Form==0, u, bc, J=Gain, \
            solver_parameters={"newton_solver":{"relative_tolerance":1e-5}},
            form_compiler_parameters={"cpp_optimize": True, \
                    "representation": "quadrature", "quadrature_degree": 2})

    file_p << (u.split()[0], t)
    file_v << (u.split()[1], t)

    II_ = project(II, FunctionSpace(mesh, 'P', 1), \
            form_compiler_parameters={"cpp_optimize": True, \
            "representation": "quadrature", "quadrature_degree": 2})
    sigma = -p * Identity(2) + fluid.apparent_viscosity(II_) * rate_of_strain(v)
    sigma_ = project(sigma, TensorFunctionSpace(mesh, 'P', 1), \
            form_compiler_parameters={"cpp_optimize": True, \
            "representation": "quadrature", "quadrature_degree": 2})

    print('%d\t %.3f\t %.3f\t %.2e' % (step, t, u(0.5, 0.5)[1], u(0.5, 0.5)[1] - u0(0.5, 0.5)[1]))
    u0.assign(u)
