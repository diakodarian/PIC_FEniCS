from __future__ import print_function
from LagrangianParticlesObject import LagrangianParticles, RandomCircle, RandomRectangle, RandomBox, RandomSphere
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
d = 3              # Space dimension
M = [32,32,30]     # Number of grid points
# Mesh dimensions: Omega = [l1, l2]X[w1, w2]X[h1, h2]
l1 = 0.
l2 = 2.*np.pi
w1 = 0.
w2 = 2.*np.pi
h1 = 0.
h2 = 2.*np.pi

if d == 2:
    mesh = Mesh("mesh/circle.xml")
elif d == 3:
    mesh = Mesh('mesh/cylinder.xml')#('mesh/sphere.xml')

File("mesh.pvd") << mesh
# mesh.init()
# from pylab import show, triplot
# fig = plt.figure()
# coords = mesh.coordinates()
# # theta goes from 0 to 2pi
# theta = np.linspace(0, 2*np.pi, 100)
#
# # the radius of the circle
# r = np.sqrt(0.25)
#
# # compute x1 and x2
# x1 = np.pi + r*np.cos(theta)
# x2 = np.pi + r*np.sin(theta)
#
# ax = fig.gca()
# ax.plot(x1, x2, c='k', linewidth=3)
# ax.set_aspect(1)
# ax.triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
# show()
# sys.exit()

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


plot(facet_f, interactive=True)
sys.exit()
# c = Cell(mesh, 1)
#
# itr_facet = SubsetIterator(facet_f, 1)
#
# facet_info = []
# facets_info = []
# for f in itr_facet:
#     facets_info.append(f)
#     facet_info.append(f.index())
#     print(c.normal(f.index(),0))
#     # print((f.midpoint().x()-np.pi)**2 + (f.midpoint().y()-np.pi)**2)
#     # for v in vertices(f):
#         # print("vertex: ", v.index())
#         # print(v.point().x(), v.point().y())
# # print("***************************************")
# # print(facet_info)
# # print(facets_info[0])
# # print("***************************************")
# #
# # for i in range(len(facets_info)):
# #     print(i, "   ", facets_info[i])
# #     print(facets_info[i])#, "  ", facets[facet_info[i]])
# # from IPython import embed; embed()
# sys.exit()
#
# Mark boundary adjacent cells
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

# Plot cell_domains
# plot(cell_domains, interactive=True)
#
# itr_cells = SubsetIterator(cell_domains, 0)
# for c in itr_cells:
#     vert = c.entities(1)
    #normals = cell(c).get_vertex_coordinates()
    #print(vert)
# # from IPython import embed; embed()
# sys.exit()
# -------------------------------Experiments------------------------------------

#-------------------------------------------------------------------------------
#                       Create object
#-------------------------------------------------------------------------------
object_type = 'spherical_object' # Options spherical_ or cylindrical_
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

# Simulation parameters:
n_pr_cell = 8        # Number of particels per cell
n_pr_super_particle = 8  # Number of particles per super particle
tot_time = 20     # Total simulation time
dt = 0.251327       # time step

n_cells = mesh.num_cells() # Number of cells
N_e = n_pr_cell*n_cells       # Number of electrons
N_i = n_pr_cell*n_cells       # Number of ions
# print "n_cells: ", n_cells

# Physical parameters
epsilon_0 = 1.       # Permittivity of vacuum
mu_0 = 1.            # Permeability of vacuum
T_e = 0.              # Temperature - electrons
T_i = 0.              # Temperature - ions
kB = 1.               # Boltzmann's constant
e = 1.                # Elementary charge
Z = 1                # Atomic number
m_e = 1.              # particle mass - electron
m_i = 1836.15267389            # particle mass - ion

alpha_e = np.sqrt(kB*T_e/m_e) # Boltzmann factor
alpha_i = np.sqrt(kB*T_i/m_i) # Boltzmann factor

q_e = -e     # Electric charge - electron
q_i = Z*e  # Electric charge - ions
w = (l2*w2)/N_e #n_pr_super_particle

capacitance_sphere = 4.*np.pi*epsilon_0*r0
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

# initial_positions = np.array([[2.50, 3.0],[2.50, 3.2]])
# initial_velocities = np.array([[1., 0.0],[0.3, .0]])
# properties = {}
# key = 'q'
# properties.setdefault(key, [])
# properties[key].append(w*q_e)
# properties[key].append(w*q_i)
#
# key = 'm'
# properties.setdefault(key, [])
# properties[key].append(w*m_e)
# properties[key].append(w*m_i)
#
# n_electrons = 1
# print(initial_positions)
# print(initial_velocities)
# print(properties)
# sys.exit()

#-------------------------------------------------------------------------------
#         Create Krylov solver
#-------------------------------------------------------------------------------
solver = PETScKrylovSolver('gmres', 'hypre_amg')#, 'hypre_amg')#'gmres', 'ilu')

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


# for vertex in vertices(mesh):
#     print(vertex.index(), "  ", vertex.x(0))
# sys.exit()

#-------------------------------------------------------------------------------
#             The capacitance of the object
#-------------------------------------------------------------------------------
c = Constant(1.0)
f = Function(V)
rho = f
bc = DirichletBC(V, c, facet_f, 1)
#boundary_values = bc.get_boundary_values()
#print("boundary_values: ", boundary_values)
phi = periodic_solver(rho, V, solver, bc)
E = E_field(phi, W)

ds = Measure('ds', domain = mesh, subdomain_data = facet_f)
n = FacetNormal(mesh)
capacitance_sphere_numerical = assemble(inner(E, -1*n)*ds(1))

print("capacitance_sphere_numerical: ", capacitance_sphere_numerical)

#
# File("rho.pvd") << rho
# File("phi.pvd") << phi
# File("E.pvd") << E
#
# sys.exit()
# #*******************************************************************************
# v2d = vertex_to_dof_map(E.function_space())
# v2d2 = vertex_to_dof_map(phi.function_space())
#
# facet_vertex = boundary_values.keys()
# dof = v2d[facet_vertex]
# print("boundary_values: ", facet_vertex)
# print("facet_dof: ", v2d[facet_vertex])
# print("E: ", E.vector()[dof])
# print("size: ", E.vector().size(), "  ", len(v2d),"  ", len(v2d2))
#
# #sys.exit()
#
# itr_facets = SubsetIterator(facet_f, 1)
# for c in itr_facets:
#     # from IPython import embed; embed()
#     vert = c.entities(0)
#     print("vert: ", vert)
#     #dof = v2d[vert]
#     #print("dof: ", dof)
#     #print(E.vector()[dof])
#
# sys.exit()
# itr_cells = SubsetIterator(cell_domains, 0)
# for c in itr_cells:
#     # from IPython import embed; embed()
#     vert = c.entities(0)
#     print("vert: ", vert)
#     dof = v2d[vert]
#     print("dof: ", dof)
#     print(E.vector()[dof])
# # print("E: ", E.vector().array())#[facet_f==1])
# plot(rho, interactive=True)
# plot(phi, interactive=True)
# plot(E, interactive=True)
# sys.exit()
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

# def extract_values(u, cell_function, subdomain_id, V):
#   dofmap = V.dofmap()
#   mesh = V.mesh()
#   for cell in cells(mesh):
#     # Preserve only the dofs in cells marked as subdomain_id
#     if cell_function[cell.index()] != subdomain_id:
#       dofs = dofmap.cell_dofs(cell.index())
#       for dof in dofs:
#         u.vector()[dof] = 0.0
#   return u
#
# u = interpolate(Expression("sin(x[0]) + cos(x[1])", degree=2), V)
# u_left  = Function(V); u_left.vector()[:]  = u.vector()
# u_left  = extract_values(u_left, facet_f, 1, V)
# ff = np.where(u_left.vector().array()!=0)[0]
# print(ff)
# plot(u, mesh, interactive=True)
# sys.exit()

Ek = []
Ep = []
t = []
object_charge = 0.0
c = Constant(0.0)
for i, step in enumerate(range(tot_time)):
    if comm.Get_rank() == 0:
        print("t: ", step)

    f = Function(V)
    rho = lp.charge_density(f)

    if periodic_field_solver:
        bc = DirichletBC(V, c, facet_f, 1)
        boundary_values = bc.get_boundary_values()
        print("boundary_values: ", boundary_values)
        phi = periodic_solver(rho, V, solver, bc)
        E = E_field(phi, W)
    else:
        phi = dirichlet_solver(rho, V, bc)
        E = E_field(phi, W)

    info = lp.step(E, i, c(0), dt=dt)
    tot_n, n_proc = lp.total_number_of_particles()
    print("total_number_of_particles: ", tot_n)
    print("   ")
    c.assign(info[3])
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

# Total energy
Et = [i + j for i, j in zip(Ep, Ek)]

if comm.Get_rank() == 0:
    if data_to_file:
        to_file.close()

    to_file = open('data/energies_nonuniform_mesh.txt', 'w')
    for i,j,k, l in zip(t, Ek, Ep, Et):
        to_file.write("%f %f %f %f\n" %(i, j, k, l))
    to_file.close()
    # lp.particle_distribution()
    plot(phi, interactive=True)
    plot(rho, interactive=True)
    plot(E, interactive=True)

    #
    # fig = plt.figure()
    # plt.plot(t,Ek, '-b')
    # plt.plot(t,Ep, '-r')
    # plt.plot(t, Et, '-g')
    # plt.show()
