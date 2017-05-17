import fenics
import pde
from settings import *
import testcases

#fenics.set_log_level(fenics.ERROR)

problem = testcases.Smooth(nx, ny)
U = problem.define_functions()
algorithm = pde.GSP(problem, alpha, nu, U)

outdir = './output/'
file_p = fenics.File(outdir + problem.name + '_' + 'pressure.pvd')
file_u = fenics.File(outdir + problem.name + '_' + 'velocity.pvd')

algorithm.solve(U)

p, u = U.split(deepcopy=True)

file_p << p
file_u << u

error_norms = problem.compute_errors(U)
assert max(max(error_norms)) < 1e-10
