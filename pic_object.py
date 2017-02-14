from __future__ import print_function
from LagrangianParticlesObject import LagrangianParticles, RandomCircle
from LagrangianParticlesObject import RandomRectangle, RandomBox, RandomSphere
from Poisson_solver import periodic_solver, dirichlet_solver, E_field
from initial_conditions import initial_conditions
from particle_distribution import speed_distribution
from mesh_types import *
from boundary_conditions import *
from particle_injection import num_particles
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
d = 3              # Space dimension
l1 = 0.            # Start position x-axis
l2 = 2.*np.pi      # End position x-axis
w1 = 0.            # Start position y-axis
w2 = 2.*np.pi      # End position y-axis
h1 = 0.            # Start position z-axis
h2 = 2.*np.pi      # End position z-axis

#-------------------------------------------------------------------------------
#                       Upload mesh
#-------------------------------------------------------------------------------
with_object = True
B_field = True
with_drift = True

object_type = 'spherical_object' # Options spherical_ or cylindrical_
if with_object:
    if d == 2:
        mesh = Mesh("mesh/circle.xml")
    elif d == 3:
        if object_type == 'spherical_object':
            mesh = Mesh('mesh/sphere.xml')
        elif object_type == 'cylindrical_object':
            mesh = Mesh('mesh/cylinder.xml')
else:
    if d == 2:
        L = [l1, w1, l2, w2]
        divisions = [32,32]
        mesh = HyperCube(L, divisions)
    if d == 3:
        L = [l1, w1, h1, l2, w2, h2]
        divisions = [32,32,32]
        mesh = HyperCube(L, divisions)
#-------------------------------------------------------------------------------
#                       Create the object
#-------------------------------------------------------------------------------
if with_object:
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

boundary_l1 = 'near((x[0]-l1), 0, tol)'
boundary_l2 = 'near((x[0]-l2), 0, tol)'
boundary_w1 = 'near((x[1]-w1), 0, tol)'
boundary_w2 = 'near((x[1]-w2), 0, tol)'
boundary_l1 = CompiledSubDomain(boundary_l1, l1=l1, tol=1E-8)
boundary_l2 = CompiledSubDomain(boundary_l2, l2=l2, tol=1E-8)
boundary_w1 = CompiledSubDomain(boundary_w1, w1=w1, tol=1E-8)
boundary_w2 = CompiledSubDomain(boundary_w2, w2=w2, tol=1E-8)
boundary_l1.mark(facet_f, 2)
boundary_l2.mark(facet_f, 3)
boundary_w1.mark(facet_f, 4)
boundary_w2.mark(facet_f, 5)

if d == 3:
    boundary_h1 = 'near((x[2]-h1), 0, tol)'
    boundary_h2 = 'near((x[2]-h2), 0, tol)'
    boundary_h1 = CompiledSubDomain(boundary_h1, h1=h1, tol=1E-8)
    boundary_h2 = CompiledSubDomain(boundary_h2, h2=h2, tol=1E-8)
    boundary_h1.mark(facet_f, 6)
    boundary_h2.mark(facet_f, 7)

#-------------------------------------------------------------------------------
#                       Index of object vertices
#-------------------------------------------------------------------------------
if with_object:
    itr_facet = SubsetIterator(facet_f, 1)
    object_vertices = set()
    for f in itr_facet:
        for v in vertices(f):
            object_vertices.add(v.index())

    object_vertices = list(object_vertices)

#-------------------------------------------------------------------------------
#                       Mark boundary adjacent cells
#-------------------------------------------------------------------------------
if with_object:
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
w = (l2*w2)/N_e  # Non-dimensionalization factor

#-------------------------------------------------------------------------------
#                   Initilize particle injection process
#-------------------------------------------------------------------------------
if d == 2:
    sigma_e = [alpha_e, alpha_e]
    sigma_i = [alpha_i, alpha_i]
    n0 = np.array([1, 0])
    n1 = np.array([-1, 0])
    n2 = np.array([0, 1])
    n3 = np.array([0, -1])
    if B_field:
        B0 = np.array([1., 0.])     # Uniform background magnetic field
    else:
        B0 = np.array([0., 0.])
    if with_drift:
        vd = np.array([1.5, 0.])  # Drift velocity of the plasma particles
        mu_e = [1.5,0.]
        mu_i = [1.5,0.]
    else:
        vd = np.array([0., 0.])
        mu_e = [0.,0.]
        mu_i = [0.,0.]
    # Normal components of velocity
    v_n0 = np.dot(vd, n0)
    v_n1 = np.dot(vd, n1)
    v_n2 = np.dot(vd, n2)
    v_n3 = np.dot(vd, n3)
    v_n = np.array([v_n0,v_n1,v_n2,v_n3])
    A_surface = l2
if d == 3:
    sigma_e = [alpha_e, alpha_e,alpha_e]
    sigma_i = [alpha_i, alpha_i,alpha_i]
    # Normal unit surface vectors
    n0 = np.array([1, 0, 0])
    n1 = np.array([-1, 0, 0])
    n2 = np.array([0, 1, 0])
    n3 = np.array([0, -1, 0])
    n4 = np.array([0, 0, 1])
    n5 = np.array([0, 0, -1])
    if B_field:
        B0 = np.array([1., 0.,0.])     # Uniform background magnetic field
    else:
        B0 = np.array([0., 0.,0.])
    if with_drift:
        vd = np.array([1.5, 0.,0.])  # Drift velocity of the plasma particles
        mu_e = [1.5,0.,0.]
        mu_i = [1.5,0.,0.]
    else:
        vd = np.array([0., 0.,0.])
        mu_e = [0.,0.,0.]
        mu_i = [0.,0.,0.]

    # Normal components of velocity
    v_n0 = np.dot(vd, n0)
    v_n1 = np.dot(vd, n1)
    v_n2 = np.dot(vd, n2)
    v_n3 = np.dot(vd, n3)
    v_n4 = np.dot(vd, n4)
    v_n5 = np.dot(vd, n5)
    v_n = np.array([v_n0,v_n1,v_n2,v_n3,v_n4,v_n5])
    A_surface = l2*w2

B0_2 = np.linalg.norm(B0)         # Norm of the B-field
E0 = -B0_2*np.cross(vd, B0)       # Induced background electric field

count_e = []
count_i = []
for i in range(len(v_n)):
    count_e.append(num_particles(A_surface, dt, n_plasma, v_n[i], sigma_e[0]))
    count_i.append(num_particles(A_surface, dt, n_plasma, v_n[i], sigma_i[0]))

capacitance_sphere = 4.*np.pi*epsilon_0*r0       # Theoretical value
print("capacitance_sphere: ", capacitance_sphere)
#-------------------------------------------------------------------------------
#                       Create boundary conditions
#-------------------------------------------------------------------------------
periodic_field_solver = False # Periodic or Dirichlet bcs
if periodic_field_solver:
    V, VV, W = periodic_bcs(mesh, L)
else:
    V = FunctionSpace(mesh, "CG", 1)
    VV = VectorFunctionSpace(mesh, "CG", 1)
    W = VectorFunctionSpace(mesh, 'DG', 0)

    phi_0 = 'x[0]*Ex + x[1]*Ey + x[2]*Ez'
    phi_l1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
    phi_l2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
    phi_w1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
    phi_w2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])

    bc0 = DirichletBC(V, phi_l1, boundary_l1)
    bc1 = DirichletBC(V, phi_l2, boundary_l2)
    bc2 = DirichletBC(V, phi_w1, boundary_w1)
    bc3 = DirichletBC(V, phi_w2, boundary_w2)

    bcs_Dirichlet = [bc0, bc1, bc2, bc3]
    if d == 3:
        phi_h1 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
        phi_h2 = Expression(phi_0, degree=1, Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])

        bc4 = DirichletBC(V, phi_h1, boundary_h1)
        bc5 = DirichletBC(V, phi_h2, boundary_h2)

        bcs_Dirichlet.append(bc4)
        bcs_Dirichlet.append(bc5)

#-------------------------------------------------------------------------------
#             Initialize particle positions and velocities
#-------------------------------------------------------------------------------
random_domain = 'box' # Options: 'sphere' or 'box'
initial_type = object_type
initial_positions, initial_velocities, properties, n_electrons = \
initial_conditions(N_e, N_i, L, w, q_e, q_i, m_e, m_i, mu_e, mu_i, sigma_e,
                   sigma_i, object_info, random_domain, initial_type)

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
lp = LagrangianParticles(VV, object_type, object_info)
lp.add_particles(initial_positions, initial_velocities, properties)

#-------------------------------------------------------------------------------
#             The capacitance of the object
#-------------------------------------------------------------------------------
if with_object:
    # Outer boundary conditions
    cap_bc0 = DirichletBC(V, Constant(0.0), boundary_l1)
    cap_bc1 = DirichletBC(V, Constant(0.0), boundary_l2)
    cap_bc2 = DirichletBC(V, Constant(0.0), boundary_w1)
    cap_bc3 = DirichletBC(V, Constant(0.0), boundary_w2)

    cap_bcs_Dirichlet = [cap_bc0, cap_bc1, cap_bc2, cap_bc3]
    if d == 3:
        cap_bc4 = DirichletBC(V, Constant(0.0), boundary_h1)
        cap_bc5 = DirichletBC(V, Constant(0.0), boundary_h2)

        cap_bcs_Dirichlet.append(cap_bc4)
        cap_bcs_Dirichlet.append(cap_bc5)

    # Object boundary value
    c = Constant(1.0)
    bc_object = DirichletBC(V, c, facet_f, 1)

    # Source term: 0 everywhere
    f = Function(V)
    phi = dirichlet_solver(f, V, cap_bcs_Dirichlet, bc_object)
    E = E_field(phi, W)

test_surface_integral = False
if test_surface_integral == True:
    # Example 1:
    # Rotational field around the circular object
    # The surface integral sould be 0.
    # func = Expression(('-1*(x[1]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))', '-1*(x[0]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))'), a=np.pi, degree=2)
    # Example 2:
    # Gravitational field on the circular/spherical object
    # The surface integral sould be 2*pi*r or 4*pi*r**2.
    if d ==2:
        func = Expression(('(x[0]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))', '(x[1]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a), 0.5))'), a=np.pi, degree=2)
    elif d == 3:
        func = Expression(('(x[0]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a)+(x[2]-a)*(x[2]-a), 0.5))', '(x[1]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a)+(x[2]-a)*(x[2]-a), 0.5))', '(x[2]-a)/(pow((x[0]-a)*(x[0]-a)+(x[1]-a)*(x[1]-a)+(x[2]-a)*(x[2]-a), 0.5))'), a=np.pi, degree=2)
    f = interpolate(func, VV)
    E.assign(f)
    plot(E, interactive=True)

if with_object:
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
    if with_object:
        phi_object = (q_object-q_rho)/capacitance_sphere_numerical
        c.assign(phi_object)

    if periodic_field_solver:
        bc_object = DirichletBC(V, c, facet_f, 1)
        # boundary_values = bc_object.get_boundary_values()
        # print("boundary_values: ", boundary_values)
        phi = periodic_solver(rho, V, solver, bc_object)
        E = E_field(phi, W)
    else:
        bc_object = DirichletBC(V, c, facet_f, 1)
        phi = dirichlet_solver(rho, V, bcs_Dirichlet, bc_object)
        E = E_field(phi, W)

    info = lp.step(E, J_e, J_i, i, q_object, dt, B0)
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
