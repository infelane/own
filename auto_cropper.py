import matplotlib.pyplot as plt
import glob
import numpy as np


def main():
    option = 'extend'   # 'black'/ 'extend'

    for filename in glob.glob('*.jpg'):  # go over all images

        print(filename)

        new_filename = filename[:-4] + '_new'+ filename[-4:]
        img = plt.imread(filename)

        shape = np.shape(img)

        h0 = shape[0]
        w0 = shape[1]

        optimal_verhouding = 7./7.5  # h/w

        h1 = int(max(h0, optimal_verhouding*w0))
        w1 = int(max(w0, h0/optimal_verhouding))

        new_shape = (h1, w1, shape[2])

        img_new = np.zeros(new_shape, dtype=np.uint8)

        diff_ver = h1-h0
        diff_hor = w1 - w0
        left = diff_hor//2
        right = diff_hor - left
        top = diff_ver//2
        bot = diff_ver - top

        img_new[top:top+h0, left:left+w0] = img

        if option == 'extend':
            img_new[:top, left:left+w0, :] = img[0:1, :, :]
            img_new[top+h0:, left:left + w0, :] = img[-1:, :, :]
            img_new[top:top+h0, :left, :] = img[:, 0:1, :]
            img_new[top:top+h0, left+w0:, :] = img[:, -1:, :]

        plt.imsave('new_images/'+new_filename, img_new)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.imshow(img_new)

    plt.show()


if __name__ == '__main__':
    main()