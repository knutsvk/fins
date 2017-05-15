# Fluid properties
rho = 1.0       # Density
mu  = 1.0       # Dynamic viscosity coefficient
ty  = 0.3       # Yield stess
eps = 1e-4      # Regularisation parameter

# Spatial discretisation
nx = ny = 32    # Number of elements per direction

# Settings for time loop
dt    = 0.05    # Time step size
t_end = 1.0     # Final time
sstol = 1e-5    # Steady-state tolerance

# Nonlinear solver settings
reltol = 1e-5   # Relative tolerance required for convergence
maxiter = 100   # Maximum allowed iterations per solve
