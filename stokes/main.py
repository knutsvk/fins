import fenics
import pde
from settings import *
import testcases

fenics.set_log_level(fenics.ERROR)


# Smooth test case with linear analytical solution
for nx, ny in [(3,3), (5,3), (3,5), (20,20), (50,50)]:
    problem = testcases.Smooth(nx, ny)
    algorithm = pde.GSP(problem, alpha, nu)
    U = problem.define_functions()
    algorithm.solve(U)
    error_norms = problem.compute_errors(U)
    assert max(max(error_norms)) < 1e-10

# Polka dot pattern, sinusoidal analytical solution
n = 128
problem = testcases.PolkaDots(nx, ny)
algorithm = pde.GSP(problem, alpha, nu)
U = problem.define_functions()
algorithm.solve(U)

error_norms = problem.compute_errors(U)
print(error_norms)

p, u = U.split(deepcopy=True)
outdir = './output/'
file_p = fenics.File(outdir + problem.name + '_' + 'pressure.pvd')
file_p << p
file_u = fenics.File(outdir + problem.name + '_' + 'velocity.pvd')
file_u << u


# Regularised lid-driven cavity 
problem = testcases.LidDrivenCavity(nx, ny)
algorithm = pde.GSP(problem, alpha, nu)
U = problem.define_functions()
algorithm.solve(U)

p, u = U.split(deepcopy=True)
file_p = fenics.File(outdir + problem.name + '_' + 'pressure.pvd')
file_p << p
file_u = fenics.File(outdir + problem.name + '_' + 'velocity.pvd')
file_u << u
