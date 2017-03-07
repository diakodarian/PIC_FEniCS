from __future__ import print_function, division
import sys
if sys.version_info.major == 2:
	from itertools import izip as zip
	range = xrange

from dolfin import *

def near(x, y, tol=1e-8):
    if abs(x-y)<tol:
        return True
    else:
        return False

class PeriodicBoundary(SubDomain):

	def __init__(self, Ld):
		SubDomain.__init__(self)
		self.Ld = Ld

	# Target domain
	def inside(self, x, onBnd):
		return bool(		any([near(a,0) for a in x])					# On any lower bound
					and not any([near(a,b) for a,b in zip(x,self.Ld)])	# But not any upper bound
					and onBnd)

	# Map upper edges to lower edges
	def map(self, x, y):
		y[:] = [a-b if near(a,b) else a for a,b in zip(x,self.Ld)]

def dirichlet_bcs(V, facet_f, n_components = 0, phi0 = Constant(0.0), E0 = None):

    d = V.mesh().geometry().dim()
    if E0 is not None:
        if d == 2:
            phi0 = 'x[0]*Ex + x[1]*Ey'
            phi0 = Expression(phi0, degree = 1, Ex = -E0[0], Ey = -E0[1])
        if d == 3:
            phi0 = 'x[0]*Ex + x[1]*Ey + x[2]*Ez'
            phi0 = Expression(phi0, degree = 1,
                              Ex = -E0[0], Ey = -E0[1], Ez = -E0[2])
    bcs = []
    for i in range(2*d):
        bc0 = DirichletBC(V, phi0, facet_f, (n_components+i+1))
        bcs.append(bc0)
    return bcs
