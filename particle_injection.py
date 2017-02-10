import numpy as np
from initial_conditions import random_1d_positions, random_2d_positions
from initial_conditions import random_velocities
from math import erf

def num_particles(A, dt, n_p, v_n, alpha):
    N = n_p*A*dt*( (alpha/(np.sqrt(2*np.pi)) * np.exp(-v_n**2/(2*alpha**2))) +\
                    0.5*v_n*(1. + erf(v_n/(alpha*np.sqrt(2)))) )
    return N

def inject_particles_2d(L, count_e, count_i, alpha_e, alpha_i, mu, sigma, dt):
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

        e_pos_y = np.concatenate([e_pos_y1, e_pos_y2])
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
                velocity = alpha_e * np.random.normal(mu, sigma, dim)
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
        random_velocities(n_electrons, n_ions, dim, alpha_e, alpha_i, mu, sigma)

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
        tot = len(index_outside)
        e_deleted= len(np.where(index_outside<=n_electrons)[0])
        n_electrons -= e_deleted
        n_ions -= (tot - e_deleted)
        p_positions = np.delete(p_positions, index_outside, axis=0)
        velocities = np.delete(velocities, index_outside, axis=0)
    return p_positions, velocities, n_electrons, n_ions

def inject_particles_3d(L, count_e, count_i, alpha_e, alpha_i, mu, sigma, dt):
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

        e_pos_z = np.concatenate([e_pos_z1, e_pos_z2])
        e_pos_y = np.concatenate([e_pos_y1, e_pos_y2])
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
                velocity = alpha_e * np.random.normal(mu, sigma, dim)
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
        random_velocities(n_electrons, n_ions, dim, alpha_e, alpha_i, mu, sigma)

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
        tot = len(index_outside)
        e_deleted= len(np.where(index_outside<=n_electrons)[0])
        n_electrons -= e_deleted
        n_ions -= (tot - e_deleted)
        p_positions = np.delete(p_positions, index_outside, axis=0)
        velocities = np.delete(velocities, index_outside, axis=0)

    return p_positions, velocities

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
    mu = 3.
    sigma = 1.
    # test 2d:
    count_e = [1, 2, 3, 4]
    count_i = [2, 3, 4, 5]
    p, vel, n_electrons, n_ions = inject_particles_2d(L2d, count_e, count_i, alpha_e, alpha_i, mu, sigma, dt)


    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca()
    skip = 1
    #n_electrons = np.sum(count_e)
    p_electrons = p[:n_electrons]
    p_ions = p[n_electrons:]

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
    # test 3d
    count_e = [2,2,3,2,2,2]
    count_i = [2,2,3,2,2,2]
    p, vel = inject_particles_3d(L3d, count_e, count_i, alpha_e, alpha_i, mu, sigma, dt)


    A = l2*l2
    dt = 0.25
    n_p = 10
    v_n = 10.0
    alpha = 1.5
    N_a = num_particles(A, dt, n_p, v_n, alpha)
    print N_a
