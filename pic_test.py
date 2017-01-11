from __future__ import print_function
from LagrangianParticles_test import LagrangianParticles, RandomCircle, RandomRectangle, RandomBox, RandomSphere
from FieldSolver import periodic_solver, dirichlet_solver
from initial_conditions import initial_conditions
from particleDistribution import speed_distribution
from mesh_types import *
import matplotlib.pyplot as plt
#from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

# Simulation parameters:
d = 2              # Space dimension
M = [10,10,20]     # Number of grid points
N_e = 20           # Number of electrons
N_i = 20           # Number of ions
tot_time = 100     # Total simulation time
dt = 0.001         # time step

# Physical parameters
rho_p = 8.*N_e       # Plasma density
T_e = 1              # Temperature - electrons
T_i = 1              # Temperature - ions
kB = 1               # Boltzmann's constant
e = 1                # Elementary charge
Z = 1                # Atomic number
m_e = 1              # particle mass - electron
m_i = 100            # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e     # Electric charge - electron
q_i = Z*e  # Electric charge - ions
w = rho_p/N_e

# Create the mesh
# Mesh dimensions: Omega = [l1, l2]X[w1, w2]X[h1, h2]
l1 = 0.
l2 = 1.
w1 = 0.
w2 = 1.
h1 = 0.
h2 = 1.
l = [l1, l2, w1, w2, h1, h2]

mesh_dimensions = 'Arbitrary_dimensions' # Options: 'Unit_dimensions' or 'Arbitrary_dimensions'
if d == 3:
    divisions = [M[0], M[1], M[2]]
    L = [l1, w1, h1, l2, w2, h2]
    if mesh_dimensions == 'Unit_dimensions':
        mesh = UnitHyperCube(divisions)
    if mesh_dimensions == 'Arbitrary_dimensions':
        mesh = HyperCube(L, divisions)
if d == 2:
    divisions = [M[0], M[1]]
    L = [l1, w1, l2, w2]
    if mesh_dimensions == 'Unit_dimensions':
        mesh = UnitHyperCube(divisions)
    if mesh_dimensions == 'Arbitrary_dimensions':
        mesh = HyperCube(L, divisions)

# Create boundary conditions
periodic_field_solver = True # Periodic or Dirichlet bcs
if periodic_field_solver:
    # Sub domain for Periodic boundary condition
    class PeriodicBoundary(SubDomain):

        def __init__(self, L):
            dolfin.SubDomain.__init__(self)
            self.Lx_left = L[0]
            self.Lx_right = L[1]
            self.Ly_left = L[2]
            self.Ly_right = L[3]
        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
            return bool((near(x[0], self.Lx_left) or near(x[1], self.Ly_left)) and
                   (not((near(x[0], self.Lx_left) and near(x[1], self.Ly_right)) or
                        (near(x[0], self.Lx_right) and near(x[1], self.Ly_left)))) and on_boundary)

        def map(self, x, y):
            if near(x[0],  self.Lx_right) and near(x[1], self.Ly_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1] - (self.Ly_right - self.Ly_left)
            elif near(x[0],  self.Lx_right):
                y[0] = x[0] - (self.Lx_right - self.Lx_left)
                y[1] = x[1]
            else:   # near(x[1], 1)
                y[0] = x[0]
                y[1] = x[1] - (self.Ly_right - self.Ly_left)

    # Create boundary and finite element
    PBC = PeriodicBoundary(l)
    V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
    V_g = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain=PBC)
    V_e = VectorFunctionSpace(mesh, 'DG', 0, constrained_domain=PBC)
else:
    # Create dolfin function spaces
    V = FunctionSpace(mesh, "CG", 1)
    V_g = VectorFunctionSpace(mesh, 'CG', 1)

    # Create Dirichlet boundary condition
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u0, boundary)

# Initial particle positions and velocities
random_domain = 'box' # Options: 'sphere' or 'box'
initial_type = 'Langmuir_waves' # 'Langmuir_waves' or 'random'
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, l, d, w, q_e, q_i, m_e, m_i,
                       alpha_e, alpha_i, random_domain, initial_type)

lp = LagrangianParticles(V_g)
lp.add_particles(initial_positions, initial_velocities, properties)

fig = plt.figure()
lp.scatter_new(fig)
fig.suptitle('Initial')

data_to_file = False

if comm.Get_rank() == 0:
    fig.show()
    if data_to_file:
        to_file = open('data.xyz', 'w')
        to_file.write("%d\n" %len(initial_positions))
        to_file.write("PIC \n")
        for p1, p2 in map(None,initial_positions[n_electrons:],
                                initial_positions[:n_electrons]):
            if d == 2:
                to_file.write("%s %f %f %f\n" %('C', p1[0], p1[1], 0.0))
                to_file.write("%s %f %f %f\n" %('O', p2[0], p2[1], 0.0))
            elif d == 3:
                to_file.write("%s %f %f %f\n" %('C', p1[0], p1[1], p1[2]))
                to_file.write("%s %f %f %f\n" %('O', p2[0], p2[1], p2[2]))
plt.ion()
save = True

Ek = []
t = []
for i, step in enumerate(range(tot_time)):
    if comm.Get_rank() == 0:
        print("t: ", step)
    f = Function(V)
    f.interpolate(Expression("0",degree=1))
    rho = lp.charge_density(f)

    if periodic_field_solver:
        phi, E = periodic_solver(rho, mesh, V, V_e)
    else:
        phi, E = dirichlet_solver(rho, V, V_g, bc)

    lp.step(E, i, dt=dt)

    # Write to file
    if data_to_file:
        lp.write_to_file(to_file)

    Ek.append(lp.energies())
    t.append(step*dt)
    lp.scatter_new(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if (save and step%1==0): plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()

if comm.Get_rank() == 0:
    if data_to_file:
        to_file.close()
    plot(phi, interactive=True)
    plot(rho, interactive=True)
    plot(E, interactive=True)
    fig = plt.figure()
    plt.plot(t,Ek)
    plt.show()
