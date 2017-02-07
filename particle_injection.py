import numpy as np
from initial_conditions import random_1d_positions, random_2d_positions
from initial_conditions import random_velocities

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


    electron_velocities, ion_velocities =\
    random_velocities(n_electrons, n_ions, dim, alpha_e, alpha_i,mu,sigma)
    velocities = []
    velocities.extend(electron_velocities)
    velocities.extend(ion_velocities)
    velocities = np.array(velocities)

    w = np.random.rand(len(velocities))
    p_positions[:,0]  += dt*w*velocities[:,0]
    p_positions[:,1]  += dt*w*velocities[:,1]

    antall = []
    print "length: ", len(p_positions)
    for j in range(len(p_positions)):
        x = p_positions[j,:]
        #print "x: ", x
        k = 2
        for i in range(dim):
            #print "    ", i,  "  ", 2*i+k
            if x[i] < L[i] or x[i] > L[2*i+k]:
                #print "out: ", x[i]
                antall.append(j)
            k -=1
    print antall, "  ", len(antall)
    return p_positions

def inject_particles_3d(L, count_e, count_i):
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
    return p_positions
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
    count_e = [1, 2, 3, 4]
    count_i = [2, 3, 4, 5]
    p = inject_particles_2d(L2d, count_e, count_i, alpha_e, alpha_i, mu, sigma, dt)
    print p

    # count_e = [2,2,3,2,2,2]
    # count_i = [2,2,3,2,2,2]
    # p = inject_particles_3d(L3d, count_e, count_i)
    # print p
