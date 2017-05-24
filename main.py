import fenics
import stokes
import problems

fenics.set_log_level(fenics.ERROR)

def run_direct_polka(n=128):
    problem = problems.PolkaDots(n, n)
    algorithm = stokes.direct(problem, 0, 1)
    U = problem.define_functions()
    algorithm.solve(U)
    problem.write_file(U, n, n)
    error_norms = problem.compute_errors(U)
    print(error_norms)

def run_direct_lid(n=128):
    problem = problems.LidDrivenCavity(n, n)
    algorithm = stokes.direct(problem, 0, 1)
    U = problem.define_functions()
    algorithm.solve(U)
    problem.write_file(U, n, n)

def test_direct_smooth():
    print('Running test_direct_smooth()')
    for nx, ny in [(3,3), (5,3), (3,5), (20,20)]:
        for nu in [0.01, 1, 100]: 
            print('%d x %d mesh, nu = %.2f' % (nx, ny, nu))
            problem = problems.Smooth(nx, ny)
            algorithm = stokes.direct(problem, 0, nu)
            U = problem.define_functions()
            algorithm.solve(U)
            error_norms = problem.compute_errors(U)
            largest_error_norm = max(max(error_norms))
            assert largest_error_norm < 1e-6, 'Largest error norm = %g' % largest_error_norm
    print('Done!')

def test_direct_polka():
    print('Running test_direct_polka()')
    for nx, ny in [(32,32), (64,64)]:
        for nu in [0.01, 1, 100]: 
            print('%d x %d mesh, nu = %.2f' % (nx, ny, nu))
            problem = problems.PolkaDots(nx, ny)
            algorithm = stokes.direct(problem, 0, 1)
            U = problem.define_functions()
            algorithm.solve(U)
            error_norms = problem.compute_errors(U)
            largest_error_norm = max(max(error_norms))
            assert largest_error_norm < 1, 'Largest error norm = %g' % largest_error_norm
    print('Done!')


if __name__ == '__main__':
    test_direct_smooth()
    test_direct_polka()
