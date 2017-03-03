from __future__ import print_function
from LagrangianParticles import LagrangianParticles
from Poisson_solver import periodic_solver, dirichlet_solver, E_field
from initial_conditions import initial_conditions
from particle_distribution import speed_distribution
from mesh_types import *
from capacitance_matrix import capacitance_matrix
from boundary_conditions import *
from mark_object import *
from get_object import *
from initialize_particle_injection import *
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import itertools
import sys

comm = pyMPI.COMM_WORLD

#-------------------------------------------------------------------------------
#                           Simulation
#-------------------------------------------------------------------------------
with_object = True
B_field = False
with_drift = True

if with_object:
    # Options spherical_ or cylindrical_ or multi_components
    object_type = 'multi_components'
    initial_type = object_type
    if B_field:
        periodic_field_solver = False     # Periodic or Dirichlet bcs
    else:
        periodic_field_solver = True # Periodic or Dirichlet bcs
else:
    object_type = None
    if B_field:
        initial_type = 'random'    # random or Langmuir_waves
        periodic_field_solver = False     # Periodic or Dirichlet bcs
    else:
        initial_type = 'Langmuir_waves'    # random or Langmuir_waves
        periodic_field_solver = True     # Periodic or Dirichlet bcs

#-------------------------------------------------------------------------------
#                       Upload mesh
#-------------------------------------------------------------------------------
if with_object:
    dim = 2
    n_components = 1
    mesh, L = mesh_with_object(dim, n_components, object_type)
    d = mesh.topology().dim()
else:
    # Mesh dimensions: Omega = [l1, l2]X[w1, w2]X[h1, h2]
    d = 2              # Space dimension
    l1 = 0.            # Start position x-axis
    l2 = 2.*np.pi      # End position x-axis
    w1 = 0.            # Start position y-axis
    w2 = 2.*np.pi      # End position y-axis
    h1 = 0.            # Start position z-axis
    h2 = 2.*np.pi      # End position z-axis

    mesh, L = simple_mesh(d, l1, l2, w1, w2, h1, h2)

#-------------------------------------------------------------------------------
#                      Get the object
#-------------------------------------------------------------------------------
if with_object:
    object_info = get_object(dim, object_type, n_components)
else:
    object_info = None

#-------------------------------------------------------------------------------
#  1) Mark facets including electrical components
#  2) Indices of object vertices
#  3) Mark boundary adjacent cells
#-------------------------------------------------------------------------------
if with_object:
    facet_f = mark_boundaries(mesh, L, object_info, n_components)
    components_cells = object_cells(mesh, facet_f, n_components)
    components_vertices = object_vertices(facet_f, n_components)
    cell_domains = mark_boundary_adjecent_cells(mesh)
else:
    components_vertices = None

# # Save sub domains to file
# file = File("Plots/domains.xml")
# file << facet_f
# # Save sub domains to VTK files
# file = File("Plots/subdomains.pvd")
# file << facet_f
# plot(facet_f,interactive=True)
#-------------------------------------------------------------------------------
#                       Simulation parameters
#-------------------------------------------------------------------------------
n_pr_cell = 8             # Number of particels per cell
n_pr_super_particle = 8   # Number of particles per super particle
tot_time = 10             # Total simulation time
dt = 0.251327             # Time step

tot_volume = assemble(1*dx(mesh)) # Volume of simulation domain

n_cells = mesh.num_cells()    # Number of cells
N_e = 10000#n_pr_cell*n_cells       # Number of electrons
N_i = 10000#n_pr_cell*n_cells       # Number of ions
num_species = 2               # Number of species
#-------------------------------------------------------------------------------
#                       Physical parameters
#-------------------------------------------------------------------------------
n_plasma = N_e/tot_volume   # Plasma density

epsilon_0 = 1.              # Permittivity of vacuum
mu_0 = 1.                   # Permeability of vacuum
T_e = 1.                    # Temperature - electrons
T_i = 1.                    # Temperature - ions
kB = 1.                     # Boltzmann's constant
e = 1.                      # Elementary charge
Z = 1                       # Atomic number
m_e = 1.                    # particle mass - electron
m_i = 1836.15267389         # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e         # Electric charge - electron
q_i = Z*e        # Electric charge - ions
w = (L[d]*L[d+1])/N_e  # Non-dimensionalization factor

Bx = 0.0; By = 0.0; Bz = 5.0;
vd_x = 0.4; vd_y = 0.0; dv_z = 0.0;

if B_field and d == 2:
    B0 = [Bx, By]
if B_field and d == 3:
    B0 = [Bx, By, Bz]
if with_drift and d == 2:
    vd = [vd_x, vd_y]
if with_drift and d == 3:
    vd = [vd_x, vd_y, vd_z]

sigma_e, sigma_i, mu_e, mu_i = [], [], [], []
for i in range(d):
    sigma_e.append(alpha_e)
    sigma_i.append(alpha_i)
    mu_e.append(vd[i])
    mu_i.append(vd[i])

if B_field:
    B0_2 = np.linalg.norm(B0)         # Norm of the B-field
    E0 = -B0_2*np.cross(vd, B0)       # Induced background electric field
    if d == 2:
        E0 = [0.1, 0.2]
#-------------------------------------------------------------------------------
#                   Initilize particle injection process
#-------------------------------------------------------------------------------
if with_drift:
    n_injected_e, n_injected_i = initialize_particle_injection(L, dt, n_plasma,
                                                           sigma_e, sigma_i, vd)
if with_object:
    r0 = 0.5
    capacitance_sphere = 4.*np.pi*epsilon_0*r0       # Theoretical value
    print("Theoretical capacitance for a sphere: ", capacitance_sphere)

#-------------------------------------------------------------------------------
#                       Create boundary conditions
#-------------------------------------------------------------------------------

if periodic_field_solver:
    V, VV, W = periodic_bcs(mesh, L)
else:
    if B_field:
        V, VV, W, bcs_Dirichlet = dirichlet_bcs_B_field(E0,
                                                        mesh,
                                                        facet_f,
                                                        n_components)

        # if d == 2:
        #     E0 = [0.1, 0.2]
        #     phi_0 = 'x[0]*Ex + x[1]*Ey'
        #     phi_l1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
        #     phi_l2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
        #     phi_w1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
        #     phi_w2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1])
        # if d == 3:
        #     phi_0 = 'x[0]*Ex + x[1]*Ey + x[2]*Ez'
        #     phi_l1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
        #     phi_l2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
        #     phi_w1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
        #     phi_w2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
        #
        # bc0 = DirichletBC(V, phi_l1, boundary_l1)
        # bc1 = DirichletBC(V, phi_l2, boundary_l2)
        # bc2 = DirichletBC(V, phi_w1, boundary_w1)
        # bc3 = DirichletBC(V, phi_w2, boundary_w2)
        #
        # bcs_Dirichlet = [bc0, bc1, bc2, bc3]
        # if d == 3:
        #     phi_h1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
        #     phi_h2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
        #
        #     bc4 = DirichletBC(V, phi_h1, boundary_h1)
        #     bc5 = DirichletBC(V, phi_h2, boundary_h2)
        #
        #     bcs_Dirichlet.append(bc4)
        #     bcs_Dirichlet.append(bc5)
    else:
        phi0 = Constant(0.0)
        V, VV, W, bcs_Dirichlet = dirichlet_bcs_zero_potential(phi0, mesh,
                                                               facet_f,
                                                               n_components)


#-------------------------------------------------------------------------------
#             Initialize particle positions and velocities
#-------------------------------------------------------------------------------
random_domain = 'box' # Options: 'sphere' or 'box'
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i, mu_e, mu_i, sigma_e,
                   sigma_i, object_info, random_domain, initial_type)

# initial_velocities[:,2] = 0.
#-------------------------------------------------------------------------------
#         Create Krylov solver
#-------------------------------------------------------------------------------
solver = PETScKrylovSolver('gmres', 'hypre_amg')

solver.parameters["absolute_tolerance"] = 1e-14
solver.parameters["relative_tolerance"] = 1e-12
solver.parameters["maximum_iterations"] = 1000
#solver.parameters["monitor_convergence"] = True
solver.parameters["convergence_norm_type"] = "true"
#for item in solver.parameters.items(): print(item)
solver.set_reuse_preconditioner(True)

#-------------------------------------------------------------------------------
#             Add particles to the mesh
#-------------------------------------------------------------------------------
lp = LagrangianParticles(VV, object_type, object_info, B_field, n_injected_e,
                         n_injected_i, mu_e, mu_i, sigma_e, sigma_i, w, q_e, q_i,
                        m_e, m_i, dt)
lp.add_particles(initial_positions, initial_velocities, properties)

#-------------------------------------------------------------------------------
#             The capacitance matrix of the object
#-------------------------------------------------------------------------------
if with_object:
    capacitance, inv_capacitance = capacitance_matrix(V,
                                                      W,
                                                      mesh,
                                                      facet_f,
                                                      n_components,
                                                      epsilon_0)

#-------------------------------------------------------------------------------
#          Circuit Components and Differential Biasing
#-------------------------------------------------------------------------------
circuits = True
if circuits:
    n_circuits = 2
    n_comps_circuit_1 = 3
    n_comps_circuit_2 = 1

    capacitance_difference = np.ones((n_comps_circuit_1,n_comps_circuit_1))
    for i in range(1,n_comps_circuit_1):
        for j in range(n_comps_circuit_1):
            capacitance_difference[i-1,j] =\
                                    inv_capacitance[i,j] - inv_capacitance[0,j]

    inv_capacitance_difference = np.linalg.inv(capacitance_difference)

    # Potential differences between components in circuit 1
    diff_epsilon1 = 0.1
    diff_epsilon2 = 0.2
    diff_potential = [diff_epsilon1, diff_epsilon2, 0.0]
    diff_potential = np.asarray(diff_potential)
    print("")
    print("Difference capacitance matrix: ")
    print("")
    print(capacitance_difference)
    print("-------------------------")
    print("")
    print("Inverse of difference capacitance matrix: ")
    print("")
    print(inv_capacitance_difference)
    print("-------------------------")
    print("")

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
        to_file = open('data/dataB.xyz', 'w')
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

Ek = []              # List to store kinetic energy
Ep = []              # List to store potential energy
t = []               # List to store time

# Initial object charge
c = []
q_object = []
for i in range(n_components):
    c.append(Constant(0.0))
    q_object.append(0.0)
# Current density
J_e = Function(VV)
J_i = Function(VV)

#-------------------------------------------------------------------------------
#             Time loop
#-------------------------------------------------------------------------------
for i, step in enumerate(range(tot_time)):
    if comm.Get_rank() == 0:
        print("t: ", step)

    # Source term (charge density)
    f = Function(V)
    rho, q_rho = lp.charge_density(f, components_vertices)

    # Objetc boundary condition
    if with_object:
        bc_object = []
        for k in range(n_components):
            phi_object = 0.0
            for j in range(n_components):
                phi_object += (q_object[j]-q_rho[j])*inv_capacitance[k,j]
            c[k].assign(phi_object)
            bc_object.append(DirichletBC(V, c[k], facet_f, k+1))
    else:
        bc_object = None
    # Solver
    if periodic_field_solver:
        # boundary_values = bc_object.get_boundary_values()
        # print("boundary_values: ", boundary_values)
        phi = periodic_solver(rho, V, solver, bc_object)
        E = E_field(phi, W)
    else:
        phi = dirichlet_solver(rho, V, bcs_Dirichlet, bc_object)
        E = E_field(phi, W)

    info = lp.step(E, J_e, J_i, i, q_object, dt, B0)
    q_object = info[3]

    # circuits:
    if circuits:
        diff_potential[2] = np.sum(q_object[:-1]) - np.sum(q_rho[:-1])
        q_object[:-1] = np.dot(inv_capacitance_difference,diff_potential) + q_rho[:-1]

    # J_e = info[4]
    # J_i = info[5]

    # J_e, J_i = lp.current_density(J_e, J_i)

    tot_n, n_proc = lp.total_number_of_particles()
    print("total_number_of_particles: ", tot_n)


    Ek.append(info[2])
    energy = lp.potential_energy(phi)
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
    print("   ")


Et = [i + j for i, j in zip(Ep, Ek)]   # Total energy

#-------------------------------------------------------------------------------
#             Post-processing
#-------------------------------------------------------------------------------
if comm.Get_rank() == 0:
    if data_to_file:
        to_file.close()

    to_file = open('data/energies.txt', 'w')
    for i,j,k, l in zip(t, Ek, Ep, Et):
        to_file.write("%f %f %f %f\n" %(i, j, k, l))
    to_file.close()
    #lp.particle_distribution()
    File("Plots/rho.pvd") << rho
    File("Plots/phi.pvd") << phi
    File("Plots/E.pvd") << E
    File("Plots/J_e.pvd") << J_e
    File("Plots/J_i.pvd") << J_i
