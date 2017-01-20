from __future__ import print_function
#from mshr import *
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI
# import mshr

comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()

def UnitHyperCube(divisions):
    mesh_classes = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
    d = len(divisions)
    mesh = mesh_classes[d-1](*divisions)
    return mesh

def HyperCube(coordinates, divisions):
    mesh_classes = [RectangleMesh, BoxMesh]
    d = len(divisions)
    mesh = mesh_classes[d-2](Point(*coordinates[:d]), Point(*coordinates[d:]),
                            *divisions)
    return mesh

# def mshr_mesh(coordinates, divisions, o, r):
#     d = len(divisions)
#     rectangle_domain = Rectangle(Point(*coordinates[:d]), Point(*coordinates[d:]))
#     spherical_object = Circle(Point(*o), r)
#     domain = rectangle_domain - spherical_object
#     mesh = generate_mesh(domain, divisions[0])
#     return mesh
#
# def test_mshr_mesh():
#     from pylab import show, triplot
#     divs = [10,10]
#     L = [0, 0, 6.28, 6.28]
#     o = [3.14, 3.14]
#     print(" sdf: ", *o)
#     r = 0.75
#     mesh = mshr_mesh(L, divs, o, r)
#     coords = mesh.coordinates()
#     triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
#     show()

def test_Unit_mesh():
    from pylab import show, triplot
    divs = [[10,10], [10,10,10]]
    for i in range(len(divs)):
        divisions = divs[i]
        mesh = UnitHyperCube(divisions)
        coords = mesh.coordinates()
        triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
        show()
        #plot(mesh, interactive = True)

def test_mesh():
    from pylab import show, triplot
    divs = [[10,10], [10,10,10]]
    L = [[-1., -1, 2., 1.], [-1., -1, 0, 2., 1., 1.]]
    for i,j in zip(L, divs):
        L = i
        divisions = j
        mesh = HyperCube(L, divisions)
        coords = mesh.coordinates()
        triplot(coords[:,0], coords[:,1], triangles=mesh.cells())
        show()
        #plot(mesh, interactive = True)

if __name__=='__main__':

    test_Unit_mesh()
    test_mesh()
    #test_mshr_mesh()
