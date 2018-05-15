import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def T(im, angle = 0):
    shape = np.shape(im)
    h, w = shape
    if angle == 0:      # sum horizontally
        matrix = np.zeros((h, w, h))
        for i_h in range(h):
            matrix[i_h, :, i_h] = 1
        flat = np.reshape(matrix, (w*h, h))
    elif angle == 90:   # sum vertically
        matrix = np.zeros((h, w, w))
        for i_w in range(w):
            matrix[:, i_w, i_w] = 1
        flat = np.reshape(matrix, (w*h, w))
    else:               # along angle
        rows, cols = h, w

        # TODO have larger range of 'summations'
        matrix = np.zeros((h, w, h))
        for i_h in range(h):
            matrix[i_h, :, i_h] = 1

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        matrix = cv2.warpAffine(matrix, M, (cols, rows))
        flat = np.reshape(matrix, (w*h, h))

    return flat


def plot_inter(im1, im2):
    # clear window
    plt.clf()

    imshow = lambda x: plt.imshow(x, vmin=0, vmax=1., cmap='gray')

    plt.subplot(1, 2, 1)
    imshow(im1)
    plt.subplot(1, 2, 2)
    imshow(im2)
    plt.show()

def main():
    print(4)
    path = 'C:/Users/demo/Documents/lameeus/javascript/example/images/fig1.jpg'
    path = 'C:/Users/lameeus/Downloads/mri.jpg'
    path = 'C:/Users/lameeus/Downloads/dak.jpg'
    path = 'C:/Users/lameeus/Downloads/dak2.png'
    im = plt.imread(path)

    # downsample
    rate = 100
    im = im[::rate, ::rate, :]

    # make greyscale
    im = np.mean(im, axis=2)

    # normalize
    vmax = np.max(im)
    if vmax <= 1.:
        ...
    elif vmax <= 255.:
        im = im/255.
    else:
        raise NotImplementedError

    shape = np.shape(im)
    h, w = shape

    flat_im = np.reshape(im, (-1, ))

    if 0:
        plt.imshow(im, vmin=0, vmax=1, cmap='gray')
        plt.show()

    n_rots = 2**8 #2**7

    def gen_t(n_rots):
        t_list = [None] * n_rots
        for i_rot in range(n_rots):
            angle_i = 180 * i_rot / n_rots
            print(angle_i)
            t_i = T(im, angle_i)
            t_list[i_rot] = t_i

        t_tot = np.concatenate(t_list, axis=1)
        dpred_dest = np.transpose(t_tot)
        return t_tot, dpred_dest

    t_tot, dpred_dest = gen_t(n_rots)

    # TODO memory error

    # # outdated
    # def precalcs(flat_im, t_tot):
    #     out = np.matmul(flat_im, t_tot)
    #     return out

    def precalcs2(im, verbose=0):
        """

        :param im:
        :param verbose: 0 if no output, 1 print angles
        :return:
        """
        def T2(im, angle = 0):

            if angle == 0:  # sum horizontally
                sum_vals = np.sum(im, axis=1)
            elif angle == 90:
                sum_vals = np.sum(im, axis=0)
            else:
                shape = np.shape(im)
                h, w = shape
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                im_rot = cv2.warpAffine(im, M, (w, h))
                sum_vals = np.sum(im_rot, axis=1)

            return sum_vals

        out = [None]*n_rots
        for i_rot in range(n_rots):
            angle_i = 180 * i_rot / n_rots
            if verbose:
                print(angle_i)
            sum_vals = T2(im, angle_i)
            out[i_rot] = sum_vals

        return np.concatenate(out, axis=0)

    # out = precalcs(flat_im, t_tot)    # outdated
    out = precalcs2(im)

    # TODO better init
    def est_init():
        im_est = np.ones(shape=shape)*0.5
        flat_est = np.reshape(im_est, (-1, ))
        return flat_est

    flat_est = est_init()

    mu = 1e-3

    beta = 0.9  # momentum

    def train(flat_est, n=2):
        """
        mse = 1/2(y - pred)**2
        delta mse = dC/dw delta w
        delta_w <= - mu dC/dw
        dC/dw = dC/dpred * dpred/d_est
        dC/dpred = -(y-pred)
        dpred/d_est = t_tot^T
        """

        # initialisation of data
        # TODO put outside?
        update = np.zeros(shape=(w * h,))
        error_prev = 0

        for _ in range(n):
            out_est = np.matmul(flat_est, t_tot)
            # out_est = precalcs2(np.reshape(flat_est, (h, w)))

            diff = out - out_est
            error = 1 / 2 * np.sum(np.square(diff))

            delta_E = error - error_prev

            print('error = {}'.format(error))

            dc_dpred = -(diff)
            norm = 1./(np.shape(t_tot)[-1])

            dc_dw = np.matmul(dc_dpred, dpred_dest)
            detla_w_i = -1 * mu * norm * dc_dw

            update_next = update*beta + (1-beta)*detla_w_i

            delta_E_expected = np.sum(dc_dw * update_next)
            print('delta E: {} vs expected: {}'.format(delta_E, delta_E_expected))

            flat_est += update_next

            error_prev = error
            update[...] = update_next    # todo, add [...] / copy

        return flat_est

    def unflatten(x):
        return np.reshape(x, shape)

    def foo(n_iter):
        train(flat_est, n_iter)
        a = unflatten(flat_est)
        plot_inter(im, a)

        return a

    foo(10)

    a = 1
    print('HERE')


if __name__ == '__main__':
    main()
