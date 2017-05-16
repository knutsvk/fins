import fenics
import pde
import rheology
from settings import *
import testcases

#fenics.set_log_level(fenics.ERROR)

problem = testcases.Smooth(nx, ny)
U = problem.define_functions()
algorithm = pde.GSP(problem, alpha, nu, U)

outdir = './output/'
file_p = fenics.File(outdir + problem.name + '_' + 'pressure.pvd')
file_v = fenics.File(outdir + problem.name + '_' + 'velocity.pvd')

algorithm.solve(U)

pressure, velocity = U.split(deepcopy=True)

file_p << (pressure)
file_v << (velocity)

# TODO: Check that p - (-x+y) = C
# TODO: Check that div(u) = 0
