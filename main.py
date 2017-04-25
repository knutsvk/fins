import fenics
import rheology
import pde
import testcases

fenics.set_log_level(fenics.ERROR)

rho = 1.0
mu = 1.0
tau_y = 0.3
eps = 1e-4
fluid = rheology.Sigmoid(rho, mu, tau_y, eps)

t = 0.0
dt = 0.05
t_end = 1.0

nx = ny = 32
problem = testcases.Poiseuille(nx, ny)

u, u0 = problem.define_functions()
algorithm = pde.Abali(problem, dt, fluid, u, u0)

outdir = './output/'
file_p = fenics.File(outdir + problem.name + '_' + fluid.name + '_' + 'pressure.pvd')
file_v = fenics.File(outdir + problem.name + '_' + fluid.name + '_' + 'velocity.pvd')

step = 0
print('step\t time\t umax\t change')
while t < t_end:
    step += 1
    t += dt
    algorithm.advance(u)

    file_p << (u.split()[0], t)
    file_v << (u.split()[1], t)

    print('%d\t %.3f\t %.3f\t %.2e' % (step, t, u(0.5, 0.5)[1], u(0.5, 0.5)[1] - u0(0.5, 0.5)[1]))
    u0.assign(u)
