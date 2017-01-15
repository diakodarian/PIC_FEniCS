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

# Simulation parameters:
d = 2              # Space dimension
M = [50,50,30]     # Number of grid points
N_e = 70           # Number of electrons
N_i = 70           # Number of ions
tot_time = 100     # Total simulation time
dt = 0.001         # time step

# Physical parameters
rho_p = 8.*N_e       # Plasma density
T_e = 0.              # Temperature - electrons
T_i = 0.              # Temperature - ions
kB = 1.               # Boltzmann's constant
e = 1.                # Elementary charge
Z = 1                # Atomic number
m_e = 1.              # particle mass - electron
m_i = 100.            # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e     # Electric charge - electron
q_i = Z*e  # Electric charge - ions
w = rho_p/N_e

#-------------------------------------------------------------------------------
#                           Create the mesh
#-------------------------------------------------------------------------------
# Mesh dimensions: Omega = [l1, l2]X[w1, w2]X[h1, h2]
l1 = 0.
l2 = 1.
w1 = 0.
w2 = 1.
h1 = 0.
h2 = 1.

mesh_dimensions = 'Unit_dimensions' # Options: 'Unit_dimensions' or 'Arbitrary_dimensions'
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

#-------------------------------------------------------------------------------
#                       Create boundary conditions
#-------------------------------------------------------------------------------
periodic_field_solver = True # Periodic or Dirichlet bcs
if periodic_field_solver:
    V, VV, V_e = periodic_bcs(mesh, L)
else:
    u_D = Constant(-6.0)
    bc, V, VV, V_e = dirichlet_bcs(u_D, mesh)

#-------------------------------------------------------------------------------
#             Initialize particle positions and velocities
#-------------------------------------------------------------------------------
random_domain = 'box' # Options: 'sphere' or 'box'
initial_type = 'Langmuir_waves' # 'Langmuir_waves' or 'random'
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i,
                       alpha_e, alpha_i, random_domain, initial_type)

# Tests:
# initial_positions = [[0.3, 0.1], [1./6., 0.9]]
# initial_velocities = [[0.,0.], [0.,0.]]
# initial_velocities = np.array(initial_velocities)
# initial_positions = np.array(initial_positions)
#
# properties = {}
# key = 'q'
# properties.setdefault(key, [])
# properties[key].append(w*q_e)
# properties[key].append(w*q_i)
#
# key = 'm'
# properties.setdefault(key, [])
# properties[key].append(m_e)
# properties[key].append(m_i)
#
# print(initial_positions)
# print(initial_velocities)
# print(properties)
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
Ep = []
t = []
for i, step in enumerate(range(tot_time)):
    if comm.Get_rank() == 0:
        print("t: ", step)


    f = Constant("0.0")
    #f = Expression("sin(2*pi*x[0])", degree=1)
    #f = Function(V)
    f = interpolate(f, V)
    # f_dofs = f.function_space().dofmap().dofs()
    # print("f_dofs: ", f_dofs)
    # print(f.vector()[f_dofs])
    # coor = f.function_space().mesh().coordinates()
    # for i, value in enumerate(f.compute_vertex_values()):
    #     print('f: vertex %d, x = %s, f = %g' %
    #           (i, tuple(coor[i]), value))
    #
    # sys.exit()


    rho = lp.charge_density(f)


    # coor = rho.function_space().mesh().coordinates()
    # for i, value in enumerate(rho.compute_vertex_values()):
    #     print('rho: vertex %d, x = %s, rho = %g' %
    #           (i, tuple(coor[i]), value))
    # sys.exit()
    if periodic_field_solver:
        phi = periodic_solver(rho, V)
        E = E_field(phi, V_e)
    else:
        phi = dirichlet_solver(rho, V, bc)
        E = E_field(phi, V_e)

    lp.step(E, i, dt=dt)

    # Write to file
    if data_to_file:
        lp.write_to_file(to_file)

    Ek.append(lp.kinetic_energy())
    Ep.append(lp.potential_energy(phi))
    t.append(step*dt)
    lp.scatter_new(fig)
    fig.suptitle('At step %d' % step)
    fig.canvas.draw()

    if (save and step%1==0): plt.savefig('img%s.png' % str(step).zfill(4))

    fig.clf()

if comm.Get_rank() == 0:
    if data_to_file:
        to_file.close()

    to_file = open('energies.txt', 'w')
    for i,j,k in zip(t, Ek, Ep):
        to_file.write("%f %f %f \n" %(i, j, k))
    to_file.close()
    # plot(phi, interactive=True)
    # plot(rho, interactive=True)
    # plot(E, interactive=True)

    fig = plt.figure()
    plt.plot(t,Ek, '-b')
    plt.plot(t,Ep, '-r')
    plt.show()
