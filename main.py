import fenics
import pde
import rheology
from settings import *
import testcases

fenics.set_log_level(fenics.ERROR)

fluid = rheology.Sigmoid(rho, mu, ty, eps)
problem = testcases.Poiseuille(nx, ny)
u, u0 = problem.define_functions()
algorithm = pde.Abali(problem, dt, fluid, u, u0)

outdir = './output/'
file_p = fenics.File(outdir + problem.name + '_' + fluid.name + '_' + 'pressure.pvd')
file_v = fenics.File(outdir + problem.name + '_' + fluid.name + '_' + 'velocity.pvd')

t = 0.0
step = 0
change = 1
print('step\t time\t vmax\t change')

while t < t_end and change > sstol:
    step += 1
    t += dt
    algorithm.advance(u)

    pressure, velocity = u.split(deepcopy=True)

    file_p << (pressure, t)
    file_v << (velocity, t)

    vmax = velocity.vector().norm('linf')
    change = (velocity.vector()-u0.split(deepcopy=True)[1].vector()).norm('linf')
    print('%d\t %.3f\t %.3f\t %.2e' % (step, t, vmax, change))

    u0.assign(u)
