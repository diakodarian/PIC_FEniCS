import numpy as np
from initial_conditions import random_1d_positions, random_2d_positions


def inject_particles_2d(L, count_e, count_i):
    l1 = L[0]
    w1 = L[1]
    l2 = L[2]
    w2 = L[3]
    Lx = [l1,l2]
    Ly = [w1,w2]
    dim = len(count_e)/2
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

    count_e = [1, 2, 3, 4]
    count_i = [2, 3, 4, 5]
    p = inject_particles_2d(L2d, count_e, count_i)
    print p

    count_e = [2,2,3,2,2,2]
    count_i = [2,2,3,2,2,2]
    p = inject_particles_3d(L3d, count_e, count_i)
    print p
