import matplotlib.pyplot as plt
import glob
import numpy as np


def main():
    option = 'extend'   # 'black'/ 'extend' / 'autocolor' / 'crop' / 'white' /'red'

    # import glob, os
    # os.chdir("ideetjes")
    for filename in glob.glob('*.png'):  # go over all images

        print(filename)

        # new_filename = filename[:-5] + '_new.png'
        new_filename = filename[:-4] + '_new.png'
        img = plt.imread(filename)*255

        shape = np.shape(img)

        h0 = shape[0]
        w0 = shape[1]

        optimal_verhouding = 7./7.5  # h/w

        if option == 'crop':
            h1 = int(min(h0, optimal_verhouding * w0))
            w1 = int(min(w0, h0 / optimal_verhouding))

        else:
            h1 = int(max(h0, optimal_verhouding * w0))
            w1 = int(max(w0, h0 / optimal_verhouding))

        new_shape = (h1, w1, shape[2])
        img_new = np.ones(new_shape, dtype=np.uint8)

        if option == 'black':
            img_new[...] = 0
        elif option == 'white':
            img_new[...] = 255
        elif option == 'red':
            img_new[..., :] = [255, 0, 0]

        if option == 'crop':

            diff_ver = h0 - h1
            diff_hor = w0 - w1
            left = diff_hor // 2
            right = diff_hor - left
            top = max(diff_ver // 2, 0)
            bot = diff_ver - top

            img_new[...] = img[top:top+h1, left:left + w1 ,:]
        else:
            diff_ver = h1 - h0
            diff_hor = w1 - w0
            left = diff_hor // 2
            right = diff_hor - left
            top = diff_ver // 2
            bot = diff_ver - top

            img_new[top:top+h0, left:left+w0] = img

        if option == 'extend':
            margin = 2
            img_new[:top, left:left+w0, :] =  np.mean(img[0:margin, :, :], axis=(0), keepdims=True)
            img_new[top+h0:, left:left + w0, :] = np.mean(img[-margin:, :, :], axis=(0), keepdims=True)
            img_new[top:top+h0, :left, :] = np.mean(img[:, 0:margin, :], axis=(1), keepdims=True)
            img_new[top:top+h0, left+w0:, :] = np.mean(img[:, -margin:, :], axis=(1), keepdims=True)
        elif option == 'autocolor':
            margin = 2
            img_new[:top, left:left + w0, :] = np.mean(img[0:margin, :, :], axis=(0, 1))
            img_new[top + h0:, left:left + w0, :] = np.mean(img[-margin:, :, :], axis=(0, 1))
            img_new[top:top + h0, :left, :] = np.mean(img[:, 0:margin, :], axis=(0, 1))
            img_new[top:top + h0, left + w0:, :] = np.mean(img[:, -margin:, :], axis=(0, 1))

        plt.imsave('new_images/'+new_filename, img_new)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.imshow(img_new)

    plt.show()


if __name__ == '__main__':
    main()