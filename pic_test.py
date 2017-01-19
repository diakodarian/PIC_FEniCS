from __future__ import print_function
from LagrangianParticles_test import LagrangianParticles, RandomCircle, RandomRectangle, RandomBox, RandomSphere
from FieldSolver import periodic_solver, dirichlet_solver, E_field
from initial_conditions import initial_conditions
from particleDistribution import speed_distribution
from mesh_types import *
from boundary_conditions import *
import matplotlib.pyplot as plt
#from dolfin import VectorFunctionSpace, interpolate, RectangleMesh, Expression, Point
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

#-------------------------------------------------------------------------------
#                           Create the mesh
#-------------------------------------------------------------------------------
d = 2              # Space dimension
M = [32,32,30]     # Number of grid points
# Mesh dimensions: Omega = [l1, l2]X[w1, w2]X[h1, h2]
l1 = 0.
l2 = 6.28#2.*np.pi
w1 = 0.
w2 = 6.28#2.*np.pi
h1 = 0.
h2 = 1.

mesh_dimensions = 'Arbitrary_dimensions' # Options: 'Unit_dimensions' or 'Arbitrary_dimensions'
if d == 3:
    divisions = [M[0], M[1], M[2]]
    if mesh_dimensions == 'Unit_dimensions':
        L = [0., 0., 0., 1., 1., 1.]
        mesh = UnitHyperCube(divisions)
    if mesh_dimensions == 'Arbitrary_dimensions':
        L = [l1, w1, h1, l2, w2, h2]
        mesh = HyperCube(L, divisions)
if d == 2:
    divisions = [M[0], M[1]]
    if mesh_dimensions == 'Unit_dimensions':
        L = [0., 0., 1., 1.]
        mesh = UnitHyperCube(divisions)
    if mesh_dimensions == 'Arbitrary_dimensions':
        L = [l1, w1, l2, w2]
        mesh = HyperCube(L, divisions)

# Simulation parameters:
n_pr_cell = 8        # Number of particels per cell
n_pr_super_particle = 8  # Number of particles per super particle
tot_time = 25     # Total simulation time
dt = 0.251327       # time step

n_cells = mesh.num_cells() # Number of cells
N_e = n_pr_cell*n_cells       # Number of electrons
N_i = n_pr_cell*n_cells       # Number of ions


# Physical parameters
epsilon_0 = 1.       # Permittivity of vacuum
mu_0 = 1.            # Permeability of vacuum
T_e = 0.              # Temperature - electrons
T_i = 0.              # Temperature - ions
kB = 1.               # Boltzmann's constant
e = 1.                # Elementary charge
Z = 1                # Atomic number
m_e = 1.              # particle mass - electron
m_i = 1836.#15267389            # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e     # Electric charge - electron
q_i = Z*e  # Electric charge - ions
w = (l2*w2)/N_e #n_pr_super_particle



#-------------------------------------------------------------------------------
#                       Create boundary conditions
#-------------------------------------------------------------------------------
periodic_field_solver = True # Periodic or Dirichlet bcs
if periodic_field_solver:
    V, VV, W = periodic_bcs(mesh, L)
else:
    u_D = Constant(-6.0)
    bc, V, VV, W = dirichlet_bcs(u_D, mesh)

#-------------------------------------------------------------------------------
#             Initialize particle positions and velocities
#-------------------------------------------------------------------------------
random_domain = 'box' # Options: 'sphere' or 'box'
initial_type = 'Langmuir_waves' # 'Langmuir_waves' or 'random'
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i,
                       alpha_e, alpha_i, random_domain, initial_type)

#-------------------------------------------------------------------------------
#             Add particles to the mesh
#-------------------------------------------------------------------------------
lp = LagrangianParticles(VV)
lp.add_particles(initial_positions, initial_velocities, properties)

#-------------------------------------------------------------------------------
#             Plot and write to file
#-------------------------------------------------------------------------------
fig = plt.figure()
lp.scatter_new(fig)
fig.suptitle('Initial')

data_to_file = True

if comm.Get_rank() == 0:
    fig.show()
    if data_to_file:
        to_file = open('data/data.xyz', 'w')
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
Ep = []
t = []
for i, step in enumerate(range(tot_time)):
    if comm.Get_rank() == 0:
        print("t: ", step)

    f = Function(V)
    #f = Constant("0.0")
    #f = interpolate(f, V)
    rho = lp.charge_density(f)
    #rho = interpolate(rho, V)

    if periodic_field_solver:
        phi = periodic_solver(rho, V)
        E = E_field(phi, W)
    else:
        phi = dirichlet_solver(rho, V, bc)
        E = E_field(phi, W)

    info = lp.step(E, i, dt=dt)
    Ek.append(info[2])
    energy = lp.energies(phi)
    Ep.append(energy)
    t.append(step*dt)
    # Write to file
    if data_to_file:
        lp.write_to_file(to_file)

    lp.scatter_new(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if (save and step%1==0): plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()

# Total energy
Et = [i + j for i, j in zip(Ep, Ek)]

if comm.Get_rank() == 0:
    if data_to_file:
        to_file.close()

    to_file = open('data/energies.txt', 'w')
    for i,j,k, l in zip(t, Ek, Ep, Et):
        to_file.write("%f %f %f %f\n" %(i, j, k, l))
    to_file.close()
    # lp.particle_distribution()
    # plot(phi, interactive=True)
    # plot(rho, interactive=True)
    # plot(E, interactive=True)
    #
    # fig = plt.figure()
    # plt.plot(t,Ek, '-b')
    # plt.plot(t,Ep, '-r')
    # plt.plot(t, Et, '-g')
    # plt.show()
