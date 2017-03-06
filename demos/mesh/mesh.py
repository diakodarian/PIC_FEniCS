from dolfin import *
import numpy as np

# Here mesh.xml is obtained by `gmsh -2 mesh.geo` and
# `dolfin-convert mesh.msh mesh.xml`
mesh = Mesh('rectangle_periodic.xml')
plot(mesh, interactive=True)
x = mesh.coordinates().reshape((-1, 2))
xmin, ymin = np.min(x, 0)
xmax, ymax = np.max(x, 0)

# Bdries
left = CompiledSubDomain('near(x[0], xmin)', xmin=xmin)
right = CompiledSubDomain('near(x[0], xmax)', xmax=xmax)
top = CompiledSubDomain('near(x[1], ymax)', ymax=ymax)
bottom = CompiledSubDomain('near(x[1], ymin)', ymin=ymin)


def periodicity_error(mesh, subdomain_to, subdomain_from, dim, tol=1E-10):
    '''See how well the periodicity of mesh boundaries is repected.'''
    x = mesh.coordinates().reshape((-1, 2))
    # Edge to vertex connectivity
    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    # Collect vertices on to and from domains
    facet_f = FacetFunction('size_t', mesh, 0)
    subdomain_from.mark(facet_f, 1)
    subdomain_to.mark(facet_f, 2)
    # The sanity check is that the number of marked facets matches
    f1s = [f.index() for f in SubsetIterator(facet_f, 1)]
    f2s = [f.index() for f in SubsetIterator(facet_f, 2)]
    if not len(f1s) == len(f2s): return -1

    # Matching
    v1s = set(sum((e2v(f1).tolist() for f1 in f1s), []))
    v2s = set(sum((e2v(f2).tolist() for f2 in f2s), []))
    # For each point on from minimize distance of points w.r.t abs norm of
    errors = []
    for v1 in v1s:
        f = lambda v2: np.abs(x[v1]-x[v2])[dim]
        e = f(min(v2s, key=f))
        if e > tol:
            return -1
        else:
            errors.append(e)
    return max(errors)

print periodicity_error(mesh, left, right, 1)
print periodicity_error(mesh, bottom, top, 0)
