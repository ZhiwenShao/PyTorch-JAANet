import numpy as np

list_path_prefix = '../data/list/'
input_land = np.loadtxt(list_path_prefix+'BP4D_combine_1_2_land.txt')

biocular = np.zeros(input_land.shape[0])

l_ocular_x = np.mean(input_land[:,np.arange(2*20-2,2*25,2)],1)
l_ocular_y = np.mean(input_land[:,np.arange(2*20-1,2*25,2)],1)
r_ocular_x = np.mean(input_land[:,np.arange(2*26-2,2*31,2)],1)
r_ocular_y = np.mean(input_land[:,np.arange(2*26-1,2*31,2)],1)
biocular = (l_ocular_x - r_ocular_x) ** 2 + (l_ocular_y - r_ocular_y) ** 2

np.savetxt(list_path_prefix+'BP4D_combine_1_2_biocular.txt', biocular, fmt='%f', delimiter='\t')