import numpy as np
from initial_conditions import random_1d_positions, random_2d_positions
from initial_conditions import random_velocities
from math import erf
from particle_distribution import speed_distribution, hist_plot
import sys

def num_particles(A, dt, n_p, v_n, alpha):
    """
    This function calculates the number of particles that needs to be injected
    through a surface of the outer boundary with area, A, based on a drifting
    Maxwellian distribution, at each time step.
    """
    N = n_p*A*dt*( (alpha/(np.sqrt(2*np.pi)) * np.exp(-v_n**2/(2*alpha**2))) +\
                    0.5*v_n*(1. + erf(v_n/(alpha*np.sqrt(2)))) )
    return N

def inject_particles_2d(L, count_e, count_i, mu_e, mu_i, sigma_e, sigma_i, dt):
    """
    This function injects plasmas particles through the 4 surfaces of the outer
    boundaries of a 2D domain (rectangular).

    L     : Position of all outer boundary surfaces
    count_: Number of particles injected through each surface
    mu_   : Drift velocity
    sigma_: Standard deviation of particle distribution
    dt    : time step
    """
    l1 = L[0]
    w1 = L[1]
    l2 = L[2]
    w2 = L[3]
    Lx = [l1,l2]
    Ly = [w1,w2]
    dim = len(count_e)/2

    n_electrons = np.sum(count_e)
    n_ions = np.sum(count_i)
    count = []
    count.append(count_e)
    count.append(count_i)
    p_positions = []

    for i in range(len(count)):
        num = count[i]
        e_pos_x1, e_pos_x2 = random_1d_positions(Lx, num[2], num[3])
        e_pos_y1, e_pos_y2 = random_1d_positions(Ly, num[0], num[1])

        if (num[0] == 0 or num[1] == 0) and not (num[0] == 0 and num[1] == 0):
            if num[0] == 0:
                e_pos_y = e_pos_y2
            else:
                e_pos_y = e_pos_y1
        else:
            e_pos_y = np.concatenate([e_pos_y1, e_pos_y2])
        if (num[2] == 0 or num[3] == 0) and not (num[2] == 0 and num[3] == 0):
            if num[2] == 0:
                e_pos_x = e_pos_x2
            else:
                e_pos_x = e_pos_x1
        else:
            e_pos_x = np.concatenate([e_pos_x1, e_pos_x2])

        x_e = np.empty((len(e_pos_x), dim))
        y_e = np.empty((len(e_pos_y), dim))

        for i in range(len(e_pos_y)):
            if i < num[0]:
                y_e[i,0] = Lx[0]
            else:
                y_e[i,0] = Lx[1]
            y_e[i,1] = e_pos_y[i]

        for i in range(len(e_pos_x)):
            if i < num[2]:
                x_e[i,1] = Ly[0]
            else:
                x_e[i,1] = Ly[1]
            x_e[i,0] = e_pos_x[i]

        e_pos = np.concatenate([x_e, y_e])
        p_positions.extend(e_pos)

    p_positions = np.array(p_positions)

    velocities = []
    reject_particles = False
    if reject_particles:
        for i in range(len(p_positions)):
            move = True
            while move:
                velocity = alpha_e * np.random.normal(mu_e[0], sigma_e[0], dim)
                w = np.random.rand()
                x = p_positions[i] + w*dt*velocity
                k = 2
                cond = []
                for j in range(dim):
                    if x[j] < L[j] or x[j] > L[2*j+k]:
                        cond.append(j)
                    k -=1
                if len(cond)>0:
                    move = True
                elif len(cond) == 0:
                    move = False
                    p_positions[i] = x
                    velocities.append(velocity)
        velocities = np.array(velocities)
    else:
        electron_velocities, ion_velocities =\
        random_velocities(n_electrons, n_ions, mu_e, mu_i, sigma_e, sigma_i)

        velocities.extend(electron_velocities)
        velocities.extend(ion_velocities)
        velocities = np.array(velocities)
        w = np.random.rand(len(velocities))
        p_positions[:,0]  += dt*w*velocities[:,0]
        p_positions[:,1]  += dt*w*velocities[:,1]

        index_outside = []
        for i in range(len(p_positions)):
            x = p_positions[i]
            k = 2
            for j in range(dim):
                if x[j] < L[j] or x[j] > L[2*j+k]:
                    index_outside.append(i)
                k -= 1

        index_outside = set(index_outside)
        index_outside = list(index_outside)
        tot = len(index_outside)
        e_deleted = len(np.where(index_outside<=n_electrons)[0])
        n_electrons -= e_deleted
        n_ions -= (tot - e_deleted)

        # pos = np.delete(p_positions, index_outside, axis=0)
        # vel = np.delete(velocities, index_outside, axis=0)
        mask = np.ones(p_positions.shape, dtype=np.bool)
        mask[index_outside] = False
        p_pos = p_positions[mask]
        p_vel = velocities[mask]
        pos = p_pos.reshape((len(p_pos)/dim,dim))
        vel = p_vel.reshape((len(p_pos)/dim,dim))
    return pos, vel, n_electrons, n_ions

def inject_particles_3d(L, count_e, count_i, mu_e, mu_i, sigma_e, sigma_i, dt):
    """
    This function injects plasmas particles through the 6 surfaces of the outer
    boundaries of a 3D domain (box).

    L     : Position of all outer boundary surfaces
    count_: Number of particles injected through each surface
    mu_   : Drift velocity
    sigma_: Standard deviation of particle distribution
    dt    : time step
    """
    l1 = L[0]
    w1 = L[1]
    h1 = L[2]
    l2 = L[3]
    w2 = L[4]
    h2 = L[5]

    Lx = [l1,l2, h1,h2]
    Ly = [w1,w2,h1,h2]
    Lz = [l1,l2,w1,w2]

    dim = len(count_e)/2
    n_electrons = np.sum(count_e)
    n_ions = np.sum(count_i)
    count = []
    count.append(count_e)
    count.append(count_i)
    p_positions = []

    for i in range(len(count)):
        num = count[i]

        e_pos_x1, e_pos_x2 = random_2d_positions([l1,h1,l2,h2], num[2], num[3], 'box')
        e_pos_y1, e_pos_y2 = random_2d_positions([w1,h1,w2,h2], num[0], num[1], 'box')
        e_pos_z1, e_pos_z2 = random_2d_positions([l1,w1,l2,w2], num[4], num[5], 'box')

        if (num[4] == 0 or num[5] == 0) and not (num[4] == 0 and num[5] == 0):
            if num[4] == 0:
                e_pos_z = e_pos_z2
            else:
                e_pos_z = e_pos_z1
        else:
            e_pos_z = np.concatenate([e_pos_z1, e_pos_z2])
        if (num[0] == 0 or num[1] == 0) and not (num[0] == 0 and num[1] == 0):
            if num[0] == 0:
                e_pos_y = e_pos_y2
            else:
                e_pos_y = e_pos_y1
        else:
            e_pos_y = np.concatenate([e_pos_y1, e_pos_y2])
        if (num[2] == 0 or num[3] == 0) and not (num[2] == 0 and num[3] == 0):
            if num[2] == 0:
                e_pos_x = e_pos_x2
            else:
                e_pos_x = e_pos_x1
        else:
            e_pos_x = np.concatenate([e_pos_x1, e_pos_x2])

        x_e = np.empty((len(e_pos_x), dim))
        y_e = np.empty((len(e_pos_y), dim))
        z_e = np.empty((len(e_pos_z), dim))

        for i in range(len(e_pos_y)):
            if i < num[0]:
                y_e[i,0] = Lx[0]
            else:
                y_e[i,0] = Lx[1]
            y_e[i,1] = e_pos_y[i, 0]
            y_e[i,2] = e_pos_y[i, 1]

        for i in range(len(e_pos_x)):
            if i < num[2]:
                x_e[i,1] = Ly[0]
            else:
                x_e[i,1] = Ly[1]
            x_e[i,0] = e_pos_x[i,0]
            x_e[i,2] = e_pos_x[i,1]

        for i in range(len(e_pos_z)):
            if i < num[4]:
                z_e[i,2] = L[2]
            else:
                z_e[i,2] = L[5]
            z_e[i,0] = e_pos_z[i,0]
            z_e[i,1] = e_pos_z[i,1]

        e_pos = np.concatenate([x_e, y_e, z_e])
        p_positions.extend(e_pos)

    p_positions = np.array(p_positions)
    velocities = []
    reject_particles = False
    if reject_particles:
        for i in range(len(p_positions)):
            move = True
            while move:
                velocity = alpha_e * np.random.normal(mu_e[0], sigma_e[0], dim)
                w = np.random.rand()
                x = p_positions[i] + w*dt*velocity
                k = 3
                cond = []
                for j in range(dim):
                    if x[j] < L[j] or x[j] > L[2*j+k]:
                        cond.append(j)
                    k -=1
                if len(cond)>0:
                    move = True
                elif len(cond) == 0:
                    move = False
                    p_positions[i] = x
                    velocities.append(velocity)
        velocities = np.array(velocities)
    else:
        electron_velocities, ion_velocities =\
        random_velocities(n_electrons, n_ions, mu_e, mu_i, sigma_e, sigma_i)

        velocities.extend(electron_velocities)
        velocities.extend(ion_velocities)
        velocities = np.array(velocities)

        w = np.random.rand(len(velocities))
        p_positions[:,0]  += dt*w*velocities[:,0]
        p_positions[:,1]  += dt*w*velocities[:,1]
        p_positions[:,2]  += dt*w*velocities[:,2]

        index_outside = []
        for i in range(len(p_positions)):
            x = p_positions[i]
            k = 3
            for j in range(dim):
                if x[j] < L[j] or x[j] > L[2*j+k]:
                    index_outside.append(i)
                k -= 1

        index_outside = set(index_outside)
        index_outside = list(index_outside)
        tot = len(index_outside)
        e_deleted = len(np.where(index_outside<=n_electrons)[0])
        n_electrons -= e_deleted
        n_ions -= (tot - e_deleted)

        mask = np.ones(p_positions.shape, dtype=np.bool)
        mask[index_outside] = False
        p_pos = p_positions[mask]
        p_vel = velocities[mask]
        pos = p_pos.reshape((len(p_pos)/dim,dim))
        vel = p_vel.reshape((len(p_pos)/dim,dim))

    return pos, vel, n_electrons, n_ions

if __name__ == '__main__':
    l1 = 0.
    l2 = 2*np.pi
    w1 = 0.
    w2 = 2*np.pi
    h1 = 0.
    h2 = 2*np.pi
    L2d = [l1, w1, l2, w2]
    L3d = [l1,w1,h1,l2,w2,h2]

    dt = 0.1
    alpha_e = 1.
    alpha_i = 1.

    test_2d = True
    test_3d = False
    test_random_velocities = False

    # Random velocities test
    if test_random_velocities:
        d = 3
        n_electrons = 10000000
        n_ions = 5000

        mu_e = [3.,1.,1.]
        mu_i = [3,0,0]
        sigma_e = [alpha_e, alpha_e, alpha_e]
        sigma_i = [alpha_i, alpha_i, alpha_i]

        e_vel, i_vel =\
        random_velocities(n_electrons, n_ions, mu_e, mu_i, sigma_e, sigma_i)

        speed_distribution(e_vel, mu_e, sigma_e)

    # 2D test:
    if test_2d:
        vd = np.array([5., 0.])  # Drift velocity of the plasma particles
        mu_e = [5.,0.]
        mu_i = [5.,0.]
        sigma_e = [alpha_e, alpha_e]
        sigma_i = [alpha_i, alpha_i]
        # Normal unit surface vectors
        n0 = np.array([1, 0])
        n1 = np.array([-1, 0])
        n2 = np.array([0, 1])
        n3 = np.array([0, -1])
        # Normal components of velocity
        v_n0 = np.dot(vd, n0)
        v_n1 = np.dot(vd, n1)
        v_n2 = np.dot(vd, n2)
        v_n3 = np.dot(vd, n3)

        v_n = np.array([v_n0,v_n1,v_n2,v_n3])

        A = l2
        n_p = 10   # plasma density
        count_e = []
        count_i = []
        for i in range(len(v_n)):
            count_e.append(num_particles(A, dt, n_p, v_n[i], sigma_e[0]))
            count_i.append(num_particles(A, dt, n_p, v_n[i], sigma_i[0]))

        print "nums: ", count_e, "   ", count_i
        count_e = [int(i) for i in count_e]
        count_i = [int(i) for i in count_i]
        print "nums: ", count_e, "   ", count_i

        #count_e = [1, 2, 3, 4]
        #count_i = [2, 3, 4, 5]
        p, vel, n_electrons, n_ions = \
        inject_particles_2d(L2d, count_e, count_i, mu_e, mu_i, sigma_e, sigma_i, dt)

        print np.sum(count_e), n_electrons, "      ", np.sum(count_i),n_ions

        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        skip = 1
        #n_electrons = np.sum(count_e)
        p_electrons = p[:n_electrons]
        p_ions = p[n_electrons:]
        print p_electrons.shape, "  shapes: ", p_ions.shape
        ax.scatter(p_ions[::skip, 0], p_ions[::skip, 1],
                   label='ions',
                   marker='o',
                   c='r',
                   edgecolor='none')
        ax.scatter(p_electrons[::skip, 0], p_electrons[::skip, 1],
                   label='electrons',
                   marker = 'o',
                   c='b',
                   edgecolor='none')
        ax.legend(loc='best')
        ax.axis([l1, l2, w1, w2])
        plt.show()

    # 3D test:
    if test_3d:
        mu_e = [0.0,0.,0.]
        mu_i = [0.0,0.,0.]
        sigma_e = [alpha_e, alpha_e, alpha_e]
        sigma_i = [alpha_i, alpha_i, alpha_i]

        vd = np.array([.0, 0., 0.])  # Drift velocity of the plasma particles

        # Normal unit surface vectors
        n0 = np.array([1, 0, 0])
        n1 = np.array([-1, 0, 0])
        n2 = np.array([0, 1, 0])
        n3 = np.array([0, -1, 0])
        n4 = np.array([0, 0, 1])
        n5 = np.array([0, 0, -1])
        # Normal components of velocity
        v_n0 = np.dot(vd, n0)
        v_n1 = np.dot(vd, n1)
        v_n2 = np.dot(vd, n2)
        v_n3 = np.dot(vd, n3)
        v_n4 = np.dot(vd, n4)
        v_n5 = np.dot(vd, n5)

        v_n = np.array([v_n0,v_n1,v_n2,v_n3,v_n4,v_n5])

        A = l2**2
        n_p = 4   # plasma density
        count_e = []
        count_i = []
        for i in range(len(v_n)):
            count_e.append(num_particles(A, dt, n_p, v_n[i], sigma_e[0]))
            count_i.append(num_particles(A, dt, n_p, v_n[i], sigma_i[0]))

        print "nums: ", count_e, "   ", count_i
        count_e = [int(i) for i in count_e]
        count_i = [int(i) for i in count_i]
        print "nums: ", count_e, "   ", count_i

        #count_e = [2,2,3,2,2,2]
        #count_i = [2,2,3,2,2,2]

        p, vel, n_electrons, n_ions =\
        inject_particles_3d(L3d, count_e, count_i, mu_e, mu_i, sigma_e, sigma_i, dt)

        # speed_distribution(vel[:n_electrons], mu_e, sigma_e)

        import matplotlib.colors as colors
        import matplotlib.cm as cmx
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = Axes3D(fig)
        skip = 1
        #n_electrons = np.sum(count_e)
        p_electrons = p[:n_electrons]
        p_ions = p[n_electrons:]
        ax.scatter(p_ions[::skip, 0], p_ions[::skip, 1], p_ions[::skip, 2],
                   label='ions',
                   marker='o',
                   c='r',
                   edgecolor='none')
        ax.scatter(p_electrons[::skip, 0], p_electrons[::skip, 1], p_electrons[::skip, 2],
                   label='electrons',
                   marker = 'o',
                   c='b',
                   edgecolor='none')
        ax.legend(loc='best')
        ax.axis([l1, l2, w1, w2])
        plt.show()
