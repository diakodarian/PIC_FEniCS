from __future__ import print_function
from LagrangianParticles_test import LagrangianParticles, RandomCircle, RandomRectangle, RandomBox, RandomSphere
from FieldSolver import periodic_solver, dirichlet_solver
from particleDistribution import speed_distribution
import matplotlib.pyplot as plt
#from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

# Simulation parameters:
d = 2              # Space dimension
M = [10, 10, 10]   # Number of grid points
N_e = 3          # Number of electrons
N_i = 3         # Number of ions
tot_time = 10  # Total simulation time
dt = 0.001     # time step

# Physical parameters
rho_p = 8.*N_e        # Plasma density
T_e = 1        # Temperature - electrons
T_i = 1        # Temperature - ions
kB = 1         # Boltzmann's constant
e = 1        # Elementary charge
Z = 1          # Atomic number
m_e = 1        # particle mass - electron
m_i = 100        # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e     # Electric charge - electron
q_i = Z*e  # Electric charge - ions
w = rho_p/N_e

# Create the mesh
if d == 3:
    mesh = UnitCubeMesh(M[0], M[1], M[2])
if d == 2:
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
random_domain = 'box' # 'sphere' or 'box'
if d == 3:
    if random_domain == 'shpere':
        initial_electon_positions = RandomSphere([0.5,0.5,0.5], 0.15).generate([N_e, N_e, N_e])
        initial_ion_positions = RandomSphere([0.5,0.5,0.5], 0.15).generate([N_i, N_i, N_i])
    elif random_domain == 'box':
        initial_electon_positions = RandomBox([0.5,0.5,0.5], [0.6,0.6,0.6]).generate([N_e, N_e, N_e])
        initial_ion_positions = RandomBox([0.5,0.5,0.5], [0.6,0.6,0.6]).generate([N_i, N_i, N_i])
if d == 2:
    if random_domain == 'shpere':
        initial_electron_positions = RandomCircle([0.5,0.5], 0.15).generate([N_e, N_e])
        initial_ion_positions = RandomCircle([0.5,0.5], 0.15).generate([N_i, N_i])
    elif random_domain == 'box':
        initial_electron_positions = RandomRectangle([0.,0.], [1.,1.]).generate([N_e, N_e])
        initial_ion_positions = RandomRectangle([0.,0.], [1.,1.]).generate([N_i, N_i])

initial_positions = []
initial_positions.extend(initial_electron_positions)
initial_positions.extend(initial_ion_positions)
initial_positions = np.array(initial_positions)

n_ions = len(initial_ion_positions)
n_electrons = len(initial_electron_positions)
n_total_particles = len(initial_positions)

if comm.Get_rank() == 0:
    print("Total number of particles: ", n_total_particles)
    print("Total number of electrons: ", n_electrons)
    print("Total number of ions: ", n_ions)
    print("Position: electrons: ", initial_electron_positions)
    print("Position: ions: ", initial_ion_positions)
    print("all positions: ", initial_positions)

# Initial Gaussian distribution of velocity components
mu, sigma = 0., 1. # mean and standard deviation
initial_electron_velocities = np.reshape(alpha_e * np.random.normal(mu, sigma, d*n_electrons),
                                (n_electrons, d))
initial_ion_velocities = np.reshape(alpha_i * np.random.normal(mu, sigma, d*n_ions),
                                (n_ions, d))

initial_velocities = []
initial_velocities.extend(initial_electron_velocities)
initial_velocities.extend(initial_ion_velocities)
initial_velocities = np.array(initial_velocities)

#speed_distribution(initial_velocities, d, alpha_e)

# Add charge of particles to properties
properties = {}
key = 'q'
properties.setdefault(key, [])
properties[key].append(w*q_e)

for i in range(n_electrons-1):
    properties.setdefault(key, [])
    properties[key].append(w*q_e)
for i in range(n_ions):
    properties.setdefault(key, [])
    properties[key].append(w*q_i)

# Add mass of particles to properties
key = 'm'
properties.setdefault(key, [])
properties[key].append(m_e)

for i in range(n_electrons-1):
    properties.setdefault(key, [])
    properties[key].append(m_e)
for i in range(n_ions):
    properties.setdefault(key, [])
    properties[key].append(m_i)

print("length of properties: ", len(properties))
print("properties: ", properties)

lp = LagrangianParticles(V_g)
lp.add_particles(initial_positions, initial_velocities, properties)

f = Function(V)

fig = plt.figure()
lp.scatter(fig)
fig.suptitle('Initial')

if comm.Get_rank() == 0:
    fig.show()

plt.ion()
save = True

Ek = []
t = []
for i, step in enumerate(range(tot_time)):
    if comm.Get_rank() == 0:
        print("t: ", step)
    rho = lp.charge_density(f)
    if periodic_field_solver:
        phi, E = periodic_solver(rho, mesh, V, V_g)
    else:
        phi, E = dirichlet_solver(rho, V, V_g, bc)

    lp.step(E, i, dt=dt)

    Ek.append(lp.energies())
    t.append(step*dt)
    lp.scatter(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if (save and step%1==0): plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()

if comm.Get_rank() == 0:
    plot(phi, interactive=True)
    plot(rho, interactive=True)
    plot(E, interactive=True)
    fig = plt.figure()
    plt.plot(t,Ek)
    plt.show()
