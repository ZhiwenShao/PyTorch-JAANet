import numpy as np

list_path_prefix = '../data/list/'

'''
example of content in 'BP4D_combine_1_2_AUoccur.txt':
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0
'''
data_prefix = 'BP4D_combine_1_2'
imgs_AUoccur = np.loadtxt(list_path_prefix + data_prefix + '_AUoccur.txt')
AUoccur_rate = np.zeros((1, imgs_AUoccur.shape[1]))
p_n_weight = np.zeros((1, imgs_AUoccur.shape[1]))

for i in range(imgs_AUoccur.shape[1]):
    AUoccur_rate[0, i] = sum(imgs_AUoccur[:,i]>0) / float(imgs_AUoccur.shape[0])
    p_n_weight[0, i] = (imgs_AUoccur.shape[0] - sum(imgs_AUoccur[:,i]>0))/ float(sum(imgs_AUoccur[:,i]>0))

AU_weight = 1.0 / AUoccur_rate
AU_weight = AU_weight / AU_weight.sum() * AU_weight.shape[1]
np.savetxt(list_path_prefix + data_prefix + '_weight.txt', AU_weight, fmt='%f', delimiter='\t')
np.savetxt(list_path_prefix + data_prefix + '_p_n_weight.txt', p_n_weight, fmt='%f', delimiter='\t')