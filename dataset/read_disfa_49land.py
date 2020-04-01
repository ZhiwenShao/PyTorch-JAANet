import numpy as np
import scipy.io as sio

reflect_66 = sio.loadmat('reflect_66.mat')
reflect_66 = reflect_66 ['reflect_66']
reflect_66 = reflect_66.reshape(reflect_66.shape[1])

# each line contains labels of 66 landmarks for an image: x1,y1,x2,y2...
all_imgs_full_land = np.loadtxt('DISFA_combine_1_2_66land.txt')

all_imgs_land=np.zeros((all_imgs_full_land.shape[0], len(reflect_66)*2))

all_imgs_land[:, 0:all_imgs_land.shape[1]:2]=all_imgs_full_land[:,2 * reflect_66 - 2]
all_imgs_land[:, 1:all_imgs_land.shape[1]:2]=all_imgs_full_land[:,2 * reflect_66 - 1]

np.savetxt('DISFA_combine_1_2_land.txt', all_imgs_land, fmt='%.5f', delimiter=' ')