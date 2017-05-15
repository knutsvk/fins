from abc import ABC, abstractmethod
from fenics import pi, sqrt, atan, exp


class Fluid(ABC):
    """
    Abstract class for representing incompressible fluids. The fluids must be initialised with a
    density, and must implement functions which return the apparent viscosity and its derivative
    given the second invariant of the rate-of-strain tensor, II. 
    """
    def __init__(self, density):
        self.rho = density

    @abstractmethod
    def apparent_viscosity(self, II):
        pass

    @abstractmethod
    def differentiated_apparent_viscosity(self, II): 
        pass


class Newtonian(Fluid):
    """
    """
    def __init__(self, density, dynamic_viscosity): 
        super().__init__(density)
        self.mu = dynamic_viscosity
        self.name = 'newtonian'

    def apparent_viscosity(self, II):
        return 2 * self.mu

    def differentiated_apparent_viscosity(self, II): 
        return 0


class Sigmoid(Fluid): 
    """
    """
    def __init__(self, density, dynamic_viscosity, yield_stress, regularisation_parameter):
        super().__init__(density)
        self.mu = dynamic_viscosity
        self.ty = yield_stress
        self.eps = regularisation_parameter
        self.name = 'sigmoid'

    def apparent_viscosity(self, II):
        return 2 * self.mu + 2 * self.ty / (pi * sqrt(II)) * atan(sqrt(II) / self.eps)

    def differentiated_apparent_viscosity(self, II): 
        return self.ty / (pi * II) \
                * (self.eps / (self.eps**2 + II) - atan(sqrt(II) / self.eps) / sqrt(II))


class Papabing(Fluid):
    """
    """
    def __init__(self, density, dynamic_viscosity, yield_stress, regularisation_parameter):
        super().__init__(density)
        self.mu = dynamic_viscosity
        self.ty = yield_stress
        self.eps = regularisation_parameter
        self.name = 'papabing'

    def apparent_viscosity(self, II):
        return 2 * self.mu + 2 * self.ty / sqrt(II) * (1 - exp(-sqrt(II) / self.eps))

    def differentiated_apparent_viscosity(self, II): 
        return self.ty / II**1.5 \
                * ((1 + sqrt(II)/self.eps) * exp(-sqrt(II) / self.eps) - 1)

