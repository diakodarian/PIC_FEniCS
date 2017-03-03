from __future__ import print_function
from Poisson_solver import dirichlet_solver, E_field
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

def zero_Dircihlet_bcs(V, facet_f, n_components):
    d = V.mesh().topology().dim()
    # Outer boundary conditions
    outer_Dirichlet_bcs = []
    for i in range(2*d):
        cap_bc0 = DirichletBC(V, Constant(0.0), facet_f, (n_components+i+1))
        outer_Dirichlet_bcs.append(cap_bc0)

    return outer_Dirichlet_bcs

def solve_E(V, W, facet_f, n_components, outer_Dirichlet_bcs):
    # Solve Laplace equation for each electrical component
    E_object = []
    for i in range(n_components):
        bc_object = []
        for j in range(n_components):
            # Object boundary value
            # 1 at i = j and 0 at the others
            if i == j:
                c = Constant(1.0)
            else:
                c = Constant(0.0)
            bc_j = DirichletBC(V, c, facet_f, j+1) # facet indexing starts at 1
            bc_object.append(bc_j)

        # Source term: 0 everywhere
        f = Function(V)
        phi = dirichlet_solver(f, V, outer_Dirichlet_bcs, bc_object)
        E = E_field(phi, W)
        E_object.append(E)

        plot(phi, interactive=True)
        File("Plots/phi_object{i}.pvd".format(i=i)) << phi
        File("Plots/E_object{i}.pvd".format(i=i)) << E
    return E_object

def capacitance_matrix(V, W, mesh, facet_f, n_components, epsilon_0):
    outer_Dirichlet_bcs = zero_Dircihlet_bcs(V, facet_f, n_components)
    E_object = solve_E(V, W, facet_f, n_components, outer_Dirichlet_bcs)

    ds = Measure('ds', domain = mesh, subdomain_data = facet_f)
    n = FacetNormal(mesh)
    capacitance = np.empty((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            capacitance[i,j] = epsilon_0*assemble(inner(E_object[j], -1*n)*ds(i+1))

    #capacitance_sphere_numerical = assemble(inner(E, -1*n)*ds(1))
    inv_capacitance = np.linalg.inv(capacitance)
    print("                               ")
    print("Capacitance matrix:            ")
    print("                               ")
    print(capacitance)
    print("-------------------------------")
    print("                               ")
    print("Inverse of capacitance matrix: ")
    print("                               ")
    print(inv_capacitance)
    print("-------------------------------")
    print("                               ")

    return capacitance, inv_capacitance


def circuits(inv_capacitance, circuits_info):
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

if __name__=='__main__':

    from mesh_types import *
    from get_object import *
    from mark_object import *
    from boundary_conditions import *
    dim = 2
    epsilon_0 = 1.0
    n_components = 1
    object_type = 'multi_components'
    mesh, L = mesh_with_object(dim, n_components, object_type)
    object_info = get_object(dim, object_type, n_components)
    facet_f = mark_boundaries(mesh, L, object_info, n_components)
    phi0 = Constant(0.0)
    V, VV, W, bcs_Dirichlet = dirichlet_bcs_zero_potential(phi0, mesh, facet_f,
                                                           n_components)
    capacitance, inv_capacitance = capacitance_matrix(V, W, mesh, facet_f,
                                                      n_components, epsilon_0)