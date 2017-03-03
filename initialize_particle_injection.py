import numpy as np
from particle_injection import num_particles

def initialize_particle_injection(L, dt, n_plasma, sigma_e, sigma_i, vd):
    d = len(L)/2
    if d == 2:
        A_surface = L[d]
    if d == 3:
        A_surface = L[d]*L[d+1]
    # The unit vector normal to outer boundary surfaces
    unit_vec = np.identity(d)
    # Normal components of drift velocity
    vd_normal = np.empty(2*d)
    s = -1 # To insure normal vectors point outward from the domain
    j = 0
    for i in range(2*d):
        si = s**i
        if np.sign(si) == -1:
            j += 1
        vd_normal[i] = np.dot(vd, si*unit_vec[i-j])

    count_e = []
    count_i = []
    for i in range(2*d):
        count_e.append(num_particles(A_surface, dt, n_plasma,
                                     vd_normal[i], sigma_e[0]))
        count_i.append(num_particles(A_surface, dt, n_plasma,
                                     vd_normal[i], sigma_i[0]))

    diff_e = [(i - int(i)) for i in count_e]
    diff_i = [(i - int(i)) for i in count_i]

    count_e = [int(i) for i in count_e]
    count_i = [int(i) for i in count_i]

    count_e[0] += int(sum(diff_e))
    count_i[0] += int(sum(diff_i))

    return count_e, count_i

if __name__=='__main__':
    l1 = 0.            # Start position x-axis
    l2 = 2.*np.pi      # End position x-axis
    w1 = 0.            # Start position y-axis
    w2 = 2.*np.pi      # End position y-axis
    h1 = 0.            # Start position z-axis
    h2 = 2.*np.pi      # End position z-axis
    L = [l1, w1, l2, w2]

    dt = 0.251327
    n_plasma = 257.922669822
    sigma_e = [1., 1.]
    sigma_i = [0.0233370311169, 0.0233370311169]

    B0 = np.array([0.0, 5.])
    vd = np.array([.4, 0.])

    n_inj_e, n_inj_i = initialize_particle_injection(L, dt, n_plasma,
                                                     sigma_e, sigma_i, vd)
    print n_inj_e
    print n_inj_i
