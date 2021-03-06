from __future__ import print_function

from dolfin import *
import numpy as np
from mesh_types import simple_mesh
import sys
sys.path.insert(0, '/home/diako/Documents/FEniCS')
from src import *
import matplotlib.pyplot as plt
from mpi4py import MPI as pyMPI
import itertools
from scipy.fftpack import fft, rfft, irfft

comm = pyMPI.COMM_WORLD

data_to_file = True
lagrange = False
random_domain = 'box'
initial_type = 'two_stream'
B0 = None
B_field = None
object_info = None
object_type = None
components_vertices = []
n_injected_e, n_injected_i = [], []


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
#                       Simulation parameters
#-------------------------------------------------------------------------------
n_pr_cell = 8             # Number of particels per cell
n_pr_super_particle = 8   # Number of particles per super particle
tot_time = 20             # Total simulation time
dt = 0.251327             # Time step

tot_volume = assemble(1*dx(mesh)) # Volume of simulation domain

n_cells = mesh.num_cells()    # Number of cells
N_e = n_pr_cell*n_cells       # Number of electrons
N_i = n_pr_cell*n_cells       # Number of ions
num_species = 2               # Number of species
#-------------------------------------------------------------------------------
#                       Physical parameters
#-------------------------------------------------------------------------------
n_plasma = N_e/tot_volume   # Plasma density

epsilon_0 = 1.              # Permittivity of vacuum
mu_0 = 1.                   # Permeability of vacuum
T_e = 0.                    # Temperature - electrons
T_i = 0.                    # Temperature - ions
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

vd_x = 10.; vd_y = 0.0; dv_z = 0.0;

if d == 2:
    vd_top = [vd_x, vd_y]
    vd_bottom = [-vd_x, vd_y]
if d == 3:
    vd_top = [vd_x, vd_y, vd_z]
    vd_bottom = [-vd_x, vd_y, vd_z]

sigma_e, sigma_i, mu_e, mu_i = [], [], [], []
mu_e_top, mu_e_bottom = [], []
for i in range(d):
    sigma_e.append(alpha_e)
    sigma_i.append(alpha_i)
    mu_e_top.append(vd_top[i])
    mu_e_bottom.append(vd_bottom[i])
    mu_i.append(0.0)

mu_e = [mu_e_top, mu_e_bottom]

#-------------------------------------------------------------------------------
#                       Create boundary conditions
#-------------------------------------------------------------------------------
# V, VV, W = periodic_bcs(mesh, L)
PBC = PeriodicBoundary([L[d],L[d+1]])
V = FunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
VV = VectorFunctionSpace(mesh, "CG", 1, constrained_domain=PBC)
W = VectorFunctionSpace(mesh, 'DG', 0, constrained_domain=PBC)
#-------------------------------------------------------------------------------
#             Initialize particle positions and velocities
#-------------------------------------------------------------------------------
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i, mu_e, mu_i, sigma_e,
                   sigma_i, object_info, random_domain, initial_type)

#-------------------------------------------------------------------------------
#         Create Krylov solver
#-------------------------------------------------------------------------------
poisson = PoissonSolverPeriodic(V)
#-------------------------------------------------------------------------------
#             Add particles to the mesh
#-------------------------------------------------------------------------------
if not lagrange:
    pop = Population(mesh)
    distr = Distributor(V, [L[d], L[d+1]])

    # Add electrons
    xs = initial_positions[:N_e]
    vs = initial_velocities[:N_e]
    q = properties['q'][0]
    m = properties['m'][0]
    pop.addParticles(xs,vs,q,m)
    # Add electrons
    xs = initial_positions[N_e:]
    vs = initial_velocities[N_e:]
    q = properties['q'][-1]
    m = properties['m'][-1]
    pop.addParticles(xs,vs,q,m)
    #-------------------------------------------------------------------------------
    #             Time loop
    #-------------------------------------------------------------------------------
    N = tot_time
    KE = np.zeros(N-1)
    PE = np.zeros(N-1)
    KE0 = kineticEnergy(pop)
    Ld = [L[d], L[d+1]]
    for n in range(1,N):
    	print("Computing timestep %d/%d"%(n,N-1))
    	rho = distr.distr(pop)
    	phi = poisson.solve(rho)
    	E = electric_field(phi, PBC)
    	PE[n-1] = potentialEnergy(pop, phi)
    	KE[n-1] = accel(pop,E,(1-0.5*(n==1))*dt)
    	movePeriodic(pop,Ld,dt)

    KE[0] = KE0

    plt.plot(KE,label="Kinetic Energy")
    plt.plot(PE,label="Potential Energy")
    plt.plot(KE+PE,label="Total Energy")
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel("Timestep")
    plt.ylabel("Normalized Energy")
    plt.show()

    plot(rho)
    plot(phi)

    ux = Constant((1,0))
    Ex = project(inner(E,ux),V)
    plot(Ex)
    interactive()

else:
    lp = LagrangianParticles(VV, object_type, object_info, B_field, n_injected_e,
                             n_injected_i, mu_e, mu_i, sigma_e, sigma_i, w, q_e, q_i,
                            m_e, m_i, dt)
    lp.add_particles(initial_positions, initial_velocities, properties)
    #-------------------------------------------------------------------------------
    #             Plot and write to file
    #-------------------------------------------------------------------------------
    fig = plt.figure()
    lp.scatter_new(fig, object_type)
    fig.suptitle('Initial')

    if comm.Get_rank() == 0:
        fig.show()
        if data_to_file:
            to_file = open('../output/data/two_stream_data/data.xyz', 'w')
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

    # Current density
    J_e = Function(VV)
    J_i = Function(VV)

    q_object = []
    #-------------------------------------------------------------------------------
    #             Time loop
    #-------------------------------------------------------------------------------
    for i, step in enumerate(range(tot_time)):
        if comm.Get_rank() == 0:
            print("t: ", step)

        # Source term (charge density)
        f = Function(V)
        rho, q_rho = lp.charge_density(f, components_vertices)
        phi = poisson.solve(rho)
        E = electric_field(phi)

        info = lp.step(E, J_e, J_i, i, q_object, dt, B0)

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

        lp.scatter_new(fig, object_type)
        fig.suptitle('At step %d' % step)
        fig.canvas.draw()

        if (save and step%1==0): plt.savefig('../output/Plots/two_stream_plots/img%s.png' % str(step).zfill(4))

        fig.clf()
        print("   ")


    Et = [i + j for i, j in zip(Ep, Ek)]   # Total energy

    #-------------------------------------------------------------------------------
    #             Post-processing
    #-------------------------------------------------------------------------------
    if comm.Get_rank() == 0:
        if data_to_file:
            to_file.close()

        to_file = open('../output/data/two_stream_data/energies.txt', 'w')
        for i,j,k, l in zip(t, Ek, Ep, Et):
            to_file.write("%f %f %f %f\n" %(i, j, k, l))
        to_file.close()
        #lp.particle_distribution()
        File("../output/Plots/two_stream_plots/two_stream_rho.pvd") << rho
        File("../output/Plots/two_stream_plots/two_stream_phi.pvd") << phi
        File("../output/Plots/two_stream_plots/two_stream_E.pvd") << E
