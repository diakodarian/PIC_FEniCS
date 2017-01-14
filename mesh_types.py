from __future__ import print_function
from dolfin import *
import numpy as np
from mpi4py import MPI as pyMPI

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
