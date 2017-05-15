import fenics
import pde
import rheology
from settings import *
import testcases

fenics.set_log_level(fenics.ERROR)

problem = testcases.Smooth(nx, ny)
u, u0 = problem.define_functions()
algorithm = pde.Stokes(problem, alpha, nu, u)

outdir = './output/'
file_p = fenics.File(outdir + problem.name + '_' + 'pressure.pvd')
file_v = fenics.File(outdir + problem.name + '_' + 'velocity.pvd')

algorithm.advance(u)

pressure, velocity = u.split(deepcopy=True)

file_p << (pressure)
file_v << (velocity)

vmax = velocity.vector().norm('linf')
change = (velocity.vector()-u0.split(deepcopy=True)[1].vector()).norm('linf')
print('%d\t %.3f\t %.3f\t %.2e' % (step, t, vmax, change))
