"""
Incompressible Navier-Stokes equations using the Incremental Pressure
Correction Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from fenics import *

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define tensor magnitude
def magnitude(tensor):
    return sqrt(inner(tensor, tensor))

# Define apparent viscosity function
def eta(mu, u):
    tau_y = 0.3
    m = 100
    eps_mag = magnitude(epsilon(u))
    return mu + tau_y * (1 - exp(-eps_mag * m)) / (eps_mag + 1e-8)

# Define stress tensor
def sigma(mu, u, p):
 #   return 2 * eta(mu, u) * epsilon(u) - p*Identity(len(u))
    return 2 * mu * epsilon(u) - p * Identity(len(u))

def define_functions(V, Q):

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    u_  = Function(V)
    p_n = Function(Q)
    p_  = Function(Q)

    return u_n, u_, p_n, p_

def variational_problem(dt, rho, mu, f, \
        u_n, p_n, u_, p_, mesh, V, Q, bcu, bcp):

    # Define constants
    dt  = Constant(dt)
    rho = Constant(rho)
    mu  = Constant(mu)
    f   = Constant((0, 0))

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define expressions used in variational forms
    U = 0.5*(u_n + u)
    n = FacetNormal(mesh)

    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / dt, v)*dx + \
         rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
       + inner(sigma(mu, U, p_n), epsilon(v))*dx \
       + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
       - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/dt)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - dt*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    return A1, A2, A3, L1, L2, L3

def advance_ipcs(A1, A2, A3, L1, L2, L3, u_, p_, bcu, bcp):

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    return u_, p_
