from LagrangianParticles_test import LagrangianParticles, RandomCircle
from FieldSolver import solver
import matplotlib.pyplot as plt
#from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

d = 2    # Space dimension
M = [64, 64]
N = 5    # Number of particles
T = 1    # Temperature
kB = 1   # Boltzmann's constant
m = 1    # particle mass
tot_time = 20 # Total simulation time
dt = 0.001      # time step
alpha = np.sqrt(kB*T/m) # Maxwellian factor

# The mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 1), M[0], M[1])

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

# Initial particle positions
initial_positions = RandomCircle([0.5, 0.75], 0.15).generate([N, N])
n_particles = len(initial_positions)
# Gaussian distribution of velocity components
mu, sigma = 0., 1. # mean and standard deviation
initial_velocities = np.reshape(alpha * np.random.normal(mu, sigma, d*n_particles),
                                (n_particles, d))

lp = LagrangianParticles(V_g)
lp.add_particles(initial_positions, initial_velocities)

f = interpolate(Expression(("0.0", "0.0"), degree=1), V_g)

fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')

if comm.Get_rank() == 0:
    fig.show()

plt.ion()
save = True

for step in range(tot_time):
    print "t: ", step
    f_2d = lp.charge_density(f)
    rho, rho_y = f_2d.split(deepcopy=True)
    phi, E = solver(rho, mesh, V, V_g)
    #plot(phi, interactive=True)
    #sys.exit()
    lp.step(E, dt=dt)

    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if (save and step%1==0): plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()

plot(f)
