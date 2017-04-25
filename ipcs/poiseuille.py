from __future__ import print_function
from fenics import *
import numpy as np
from ipcs import define_functions, variational_problem, advance_ipcs

T = 10.0            # final time
num_steps = 500     # number of time steps
dt = T / num_steps  # time step size
rho = 1             # density
mu = 1              # kinematic viscosity
f = (0, 0)          # external body force
Nx = Ny = 16        # spatial discretisation

# Define domain mesh and corresponding function spaces
mesh = UnitSquareMesh(Nx, Ny)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define functions for solutions at previous and current time steps
u_n, u_, p_n, p_ = define_functions(V, Q)

# Define boundaries
inflow  = 'near(x[0], 0)'
outflow = 'near(x[0], 1)'
walls   = 'near(x[1], 0) || near(x[1], 1)'

# Define boundary conditions
bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow  = DirichletBC(Q, Constant(1), inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_noslip]
bcp = [bcp_inflow, bcp_outflow]

# Define variational problem and assemble matrices
A1, A2, A3, L1, L2, L3 = variational_problem(dt, rho, mu, f, \
        u_n, p_n, u_, p_, mesh, V, Q, bcu, bcp)

# Create VTK file for saving solution
vtkfile = File('output/velocity_ipcs.pvd')

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Advance to next time step
    u_, p_ = advance_ipcs(A1, A2, A3, L1, L2, L3, u_, p_, bcu, bcp)

    # Save to file and plot solution
    vtkfile << (u_, t)

    # Compute error
    u_e = Expression(('0.5*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector().array() - u_.vector().array()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().array().max())

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)
