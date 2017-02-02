from __future__ import print_function
from LagrangianParticlesObject import LagrangianParticles, RandomCircle
from LagrangianParticlesObject import RandomRectangle, RandomBox, RandomSphere
from FieldSolver import periodic_solver, dirichlet_solver, E_field
from initial_conditions import initial_conditions
from particleDistribution import speed_distribution
from mesh_types import *
from boundary_conditions import *
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

#-------------------------------------------------------------------------------
#                           Mesh parameters
#-------------------------------------------------------------------------------
# Mesh dimensions: Omega = [l1, l2]X[w1, w2]X[h1, h2]
d = 2              # Space dimension
l1 = 0.            # Start position x-axis
l2 = 2.*np.pi      # End position x-axis
w1 = 0.            # Start position y-axis
w2 = 2.*np.pi      # End position y-axis
h1 = 0.            # Start position z-axis
h2 = 2.*np.pi      # End position z-axis

#-------------------------------------------------------------------------------
#                       Upload mesh
#-------------------------------------------------------------------------------
object_type = 'spherical_object' # Options spherical_ or cylindrical_
if d == 2:
    mesh = Mesh("mesh/circle.xml")
elif d == 3:
    if object_type == 'spherical_object':
        mesh = Mesh('mesh/sphere.xml')
    elif object_type == 'cylindrical_object':
        mesh = Mesh('mesh/cylinder.xml')
#-------------------------------------------------------------------------------
#                       Create the object
#-------------------------------------------------------------------------------
if object_type == 'spherical_object':
    x0 = np.pi
    y0 = np.pi
    z0 = np.pi
    r0 = 0.5
    if d == 2:
        object_info = [x0, y0, r0]
        L = [l1, w1, l2, w2]
    elif d == 3:
        object_info = [x0, y0, z0, r0]
        L = [l1, w1, h1, l2, w2, h2]
if object_type == 'cylindrical_object':
    x0 = np.pi
    y0 = np.pi
    r0 = 0.5
    h0 = 1.0
    object_info = [x0, y0, r0, h0]
    L = [l1, w1, h1, l2, w2, h2]

#-------------------------------------------------------------------------------
#                       Mark facets
#-------------------------------------------------------------------------------
facet_f = FacetFunction('size_t', mesh, 0)
DomainBoundary().mark(facet_f, 1)
if d == 2:
    square_boundary = 'near(x[0]*(x[0]-l2), 0, tol) || near(x[1]*(x[1]-w2), 0, tol)'
    square_boundary = CompiledSubDomain(square_boundary, l2=l2, w2=w2, tol=1E-8)
    square_boundary.mark(facet_f, 2)
if d == 3:
    box_boundary = 'near(x[0]*(x[0]-l2), 0, tol) || near(x[1]*(x[1]-w2), 0, tol) || near(x[2]*(x[2]-h2), 0, tol)'
    box_boundary = CompiledSubDomain(box_boundary, l2=l2, w2=w2, h2 = h2, tol=1E-8)
    box_boundary.mark(facet_f, 2)

#-------------------------------------------------------------------------------
#                       Index of object vertices
#-------------------------------------------------------------------------------
itr_facet = SubsetIterator(facet_f, 1)
object_vertices = set()
for f in itr_facet:
    for v in vertices(f):
        object_vertices.add(v.index())

object_vertices = list(object_vertices)

#-------------------------------------------------------------------------------
#                       Mark boundary adjacent cells
#-------------------------------------------------------------------------------
if d == 2:
    boundary_adjacent_cells = [myCell for myCell in cells(mesh)
                               if any([((myFacet.midpoint().x()-np.pi)**2 + \
                                (myFacet.midpoint().y()-np.pi)**2 < 0.25) \
                                for myFacet in facets(myCell)])]
elif d == 3:
    boundary_adjacent_cells = [myCell for myCell in cells(mesh)
                               if any([((myFacet.midpoint().x()-np.pi)**2 + \
                                (myFacet.midpoint().y()-np.pi)**2 + \
                                (myFacet.midpoint().z()-np.pi)**2 < 0.25) \
                                for myFacet in facets(myCell)])]

cell_domains = CellFunction('size_t', mesh)
cell_domains.set_all(1)
for myCell in boundary_adjacent_cells:
    cell_domains[myCell] = 0

#-------------------------------------------------------------------------------
#                       Simulation parameters
#-------------------------------------------------------------------------------
n_pr_cell = 8             # Number of particels per cell
n_pr_super_particle = 8   # Number of particles per super particle
tot_time = 20             # Total simulation time
dt = 0.251327             # Time step

n_cells = mesh.num_cells()    # Number of cells
N_e = n_pr_cell*n_cells       # Number of electrons
N_i = n_pr_cell*n_cells       # Number of ions
#-------------------------------------------------------------------------------
#                       Physical parameters
#-------------------------------------------------------------------------------
epsilon_0 = 1.         # Permittivity of vacuum
mu_0 = 1.              # Permeability of vacuum
T_e = 0.               # Temperature - electrons
T_i = 0.               # Temperature - ions
kB = 1.                # Boltzmann's constant
e = 1.                 # Elementary charge
Z = 1                  # Atomic number
m_e = 1.               # particle mass - electron
m_i = 1836.15267389    # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e         # Electric charge - electron
q_i = Z*e        # Electric charge - ions
w = (l2*w2)/N_e  # Non-dimensionalization factor

capacitance_sphere = 4.*np.pi*epsilon_0*r0       # Theoretical value
print("capacitance_sphere: ", capacitance_sphere)
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
initial_type = object_type
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i,
                       alpha_e, alpha_i, object_info, random_domain, initial_type)

#-------------------------------------------------------------------------------
#         Create Krylov solver
#-------------------------------------------------------------------------------
solver = PETScKrylovSolver('gmres', 'hypre_amg')

solver.parameters["absolute_tolerance"] = 1e-14
solver.parameters["relative_tolerance"] = 1e-12
solver.parameters["maximum_iterations"] = 1000
#solver.parameters["monitor_convergence"] = True
solver.parameters["convergence_norm_type"] = "true"
#solver.parameters['preconditioner']['reuse'] = True
#solver.parameters['preconditioner']['structure'] = 'same'
#for item in solver.parameters.items(): print(item)
solver.set_reuse_preconditioner(True)

#-------------------------------------------------------------------------------
#             Add particles to the mesh
#-------------------------------------------------------------------------------
lp = LagrangianParticles(VV, object_type, object_info)
lp.add_particles(initial_positions, initial_velocities, properties)

#-------------------------------------------------------------------------------
#             The capacitance of the object
#-------------------------------------------------------------------------------
c = Constant(1.0)
f = Function(V)
bc = DirichletBC(V, c, facet_f, 1)
phi = periodic_solver(f, V, solver, bc)
E = E_field(phi, W)

test_surface_integral = False
if test_surface_integral == True:
    # Example 1:
    # Rotational field around the circular object
    # The surface integral sould be 0.
    func = Expression(('-1*(x[1]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))', '-1*(x[0]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))'), a=np.pi, degree=2)
    # Example 2:
    # Gravitational field on the circular object
    # The surface integral sould be 2*pi*r.
    # func = Expression(('(x[0]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))', '(x[1]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))'), a=np.pi, degree=2)
    f = interpolate(func, VV)
    E.assign(f)
    plot(E, interactive=True)

ds = Measure('ds', domain = mesh, subdomain_data = facet_f)
n = FacetNormal(mesh)
capacitance_sphere_numerical = assemble(inner(E, -1*n)*ds(1))

print("capacitance_sphere_numerical: ", capacitance_sphere_numerical)

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

Ek = []              # List to store kinetic energy
Ep = []              # List to store potential energy
t = []               # List to store time
q_object = 0.0       # Initial object charge
c = Constant(0.0)    # Initial object charge

J_e = Function(VV)
J_i = Function(VV)

#-------------------------------------------------------------------------------
#             Time loop
#-------------------------------------------------------------------------------
for i, step in enumerate(range(tot_time)):
    if comm.Get_rank() == 0:
        print("t: ", step)

    f = Function(V)
    rho, q_rho = lp.charge_density(f, object_vertices)

    # Objetc boundary condition
    phi_object = (q_object-q_rho)/capacitance_sphere_numerical
    c.assign(phi_object)

    if periodic_field_solver:
        bc = DirichletBC(V, c, facet_f, 1)
        boundary_values = bc.get_boundary_values()
        # print("boundary_values: ", boundary_values)
        phi = periodic_solver(rho, V, solver, bc)
        E = E_field(phi, W)
    else:
        phi = dirichlet_solver(rho, V, bc)
        E = E_field(phi, W)

    info = lp.step(E, J_e, J_i, i, q_object, dt=dt)
    q_object = info[3]
    J_e = info[4]
    J_i = info[5]

    J_e, J_i = lp.current_density(J_e, J_i)

    tot_n, n_proc = lp.total_number_of_particles()
    print("total_number_of_particles: ", tot_n)
    print("   ")

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
