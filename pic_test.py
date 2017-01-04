from LagrangianParticles_test import LagrangianParticles, RandomCircle
from FieldSolver import periodic_solver, dirichlet_solver
import matplotlib.pyplot as plt
#from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD
# Simulation parameters:
d = 2          # Space dimension
M = [64, 64]   # Number of grid points
N = 10          # Number of particles
tot_time = 20  # Total simulation time
dt = 0.001     # time step
# Physical parameters
T_e = 1        # Temperature - electrons
T_i = 1        # Temperature - ions
kB = 1         # Boltzmann's constant
q_e = 1        # Elementary charge
Z = 1          # Atomic number
m_e = 1        # particle mass - electron
m_i = 1        # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_i = Z*q_e  # Electric charge - ions


# Create the mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 1), M[0], M[1])
# Create boundary conditions
periodic_field_solver = False # Periodic or Dirichlet bcs
if periodic_field_solver:
    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
            return bool((near(x[0], 0) or near(x[1], 0)) and
                   (not((near(x[0], 0) and near(x[1], 1)) or
                        (near(x[0], 1) and near(x[1], 0)))) and on_boundary)

        def map(self, x, y):
            if near(x[0], 1) and near(x[1], 1):
                y[0] = x[0] - 1.
                y[1] = x[1] - 1.
            elif near(x[0], 1):
                y[0] = x[0] - 1.
                y[1] = x[1]
            else:   # near(x[1], 1)
                y[0] = x[0]
                y[1] = x[1] - 1.

    # Create boundary and finite element
    PBC = PeriodicBoundary()
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    V_g = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=PBC)
else:
    # Create dolfin function spaces
    V = FunctionSpace(mesh, "CG", 1)
    V_g = VectorFunctionSpace(mesh, 'CG', 1)

    # Create Dirichlet boundary condition
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, boundary)

# Initial particle positions
initial_positions = RandomCircle([0.5, 0.5], 0.15).generate([N, N])
n_particles = len(initial_positions)
if comm.Get_rank() == 0:
    print "n_particles: ", n_particles
# Initial Gaussian distribution of velocity components
mu, sigma = 0., 1. # mean and standard deviation
initial_velocities = np.reshape(alpha_e * np.random.normal(mu, sigma, d*n_particles),
                                (n_particles, d))

lp = LagrangianParticles(V_g)
lp.add_particles(initial_positions, initial_velocities)

f = Function(V)

fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')

if comm.Get_rank() == 0:
    fig.show()

plt.ion()
save = True

for step in range(tot_time):
    if comm.Get_rank() == 0:
        print "t: ", step
    rho = lp.charge_density(f)
    if periodic_field_solver:
        phi, E = periodic_solver(rho, mesh, V, V_g)
    else:
        phi, E = dirichlet_solver(rho, V, V_g, bc)

    lp.step(E, dt=dt)

    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if (save and step%1==0): plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()
if comm.Get_rank() == 0:
    plot(phi, interactive=True)
    plot(rho, interactive=True)
