from __future__ import print_function
from Poisson_solver import dirichlet_solver, electric_field
from boundary_conditions import dirichlet_bcs_zero_potential
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
import sys

comm = pyMPI.COMM_WORLD

def solve_electric_field(V, W, facet_f, n_components, outer_Dirichlet_bcs):
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
        E = electric_field(phi, W)
        E_object.append(E)

    return E_object

def capacitance_matrix(V, W, mesh, facet_f, n_components, epsilon_0):
    outer_Dirichlet_bcs = dirichlet_bcs_zero_potential(V, facet_f, n_components)
    E_object = solve_electric_field(V, W, facet_f, n_components, outer_Dirichlet_bcs)

    ds = Measure('ds', domain = mesh, subdomain_data = facet_f)
    n = FacetNormal(mesh)
    capacitance = np.empty((n_components, n_components))
    for i in range(n_components):
        for j in range(n_components):
            capacitance[i,j] = epsilon_0*assemble(inner(E_object[j], -1*n)*ds(i+1))

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

    tmp1 = []
    tmp2 = []
    for i in range(len(circuits_info)):
        if len(circuits_info[i]) > 1:
            tmp1.append(len(circuits_info[i]))
            tmp2.append(i)
    D_matrices = []
    for i in range(len(tmp1)):
        D_matrices.append(np.ones((tmp1[i], tmp1[i])))
    for i in range(len(D_matrices)):
        circuit = circuits_info[tmp2[i]]
        n_comp = len(circuit)
        for k in range(1, n_comp):
            for l in range(n_comp):
                D_matrices[i][k-1,l] = \
                                 inv_capacitance[circuit[k]-1, circuit[l]-1] -\
                                 inv_capacitance[circuit[0]-1, circuit[l]-1]

    inv_D_matrices = []
    for i in range(len(D_matrices)):
        inv_D_matrices.append(np.linalg.inv(D_matrices[i]))

    print("                                          ")
    print("Difference capacitance matrices:          ")
    print("                                          ")
    print(D_matrices)
    print("------------------------------------------")
    print("                                          ")
    print("Inverse of difference capacitance matrix: ")
    print("                                          ")
    print(inv_D_matrices)
    print("------------------------------------------")
    print("                                          ")
    return D_matrices, inv_D_matrices

if __name__=='__main__':

    from mark_object import *
    import sys
    sys.path.insert(0, '/home/diako/Documents/FEniCS/demos')
    from get_object import *

    dim = 2
    epsilon_0 = 1.0
    n_components = 4
    object_type = 'multi_circles'

    circuits_info = [[1, 3], [2, 4]]

    mesh = Mesh("demos/mesh/circuit.xml")
    V = FunctionSpace(mesh, "CG", 1)
    VV = VectorFunctionSpace(mesh, "CG", 1)
    W = VectorFunctionSpace(mesh, 'DG', 0)

    d = mesh.topology().dim()
    L = np.empty(2*d)
    for i in range(d):
        l_min = mesh.coordinates()[:,i].min()
        l_max = mesh.coordinates()[:,i].max()
        L[i] = l_min
        L[d+i] = l_max
    object_info = get_object(dim, object_type, n_components)
    facet_f = mark_boundaries(mesh, L, object_type, object_info, n_components)
    capacitance, inv_capacitance = capacitance_matrix(V, W, mesh, facet_f,
                                                      n_components, epsilon_0)

    D, inv_D = circuits(inv_capacitance, circuits_info)
