import cv2
import numpy as np
import os
import math


def align_face_49pts(img, img_land, box_enlarge, img_size):
    leftEye0 = (img_land[2 * 19] + img_land[2 * 20] + img_land[2 * 21] + img_land[2 * 22] + img_land[2 * 23] +
                img_land[2 * 24]) / 6.0
    leftEye1 = (img_land[2 * 19 + 1] + img_land[2 * 20 + 1] + img_land[2 * 21 + 1] + img_land[2 * 22 + 1] +
                img_land[2 * 23 + 1] + img_land[2 * 24 + 1]) / 6.0
    rightEye0 = (img_land[2 * 25] + img_land[2 * 26] + img_land[2 * 27] + img_land[2 * 28] + img_land[2 * 29] +
                 img_land[2 * 30]) / 6.0
    rightEye1 = (img_land[2 * 25 + 1] + img_land[2 * 26 + 1] + img_land[2 * 27 + 1] + img_land[2 * 28 + 1] +
                 img_land[2 * 29 + 1] + img_land[2 * 30 + 1]) / 6.0
    deltaX = (rightEye0 - leftEye0)
    deltaY = (rightEye1 - leftEye1)
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])

    mat2 = np.mat([[leftEye0, leftEye1, 1], [rightEye0, rightEye1, 1], [img_land[2 * 13], img_land[2 * 13 + 1], 1],
                   [img_land[2 * 31], img_land[2 * 31 + 1], 1], [img_land[2 * 37], img_land[2 * 37 + 1], 1]])

    mat2 = (mat1 * mat2.T).T

    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5

    if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))

    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
    mat = mat3 * mat1

    aligned_img = cv2.warpAffine(img, mat[0:2, :], (img_size, img_size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))

    land_3d = np.ones((int(len(img_land)/2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land)/2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.reshape(new_land[:, 0:2], len(img_land))

    return aligned_img, new_land


list_path_prefix = '../data/list/'
write_path_prefix = '../data/imgs/'
box_enlarge = 2.9
img_size = 200

all_imgs_path = open(list_path_prefix + 'BP4D_combine_1_2_path.txt').readlines()
all_imgs_land = np.loadtxt('BP4D_combine_1_2_land.txt')
# Make the landmarks be indexed from 0
all_imgs_land = all_imgs_land - 1

if not os.path.exists(write_path_prefix):
  os.makedirs(write_path_prefix)

all_imgs_new_land = np.zeros(all_imgs_land.shape)
# uncomment this for all images:
# for i in range(len(all_imgs_path)):
for i in range(2):
    full_path = all_imgs_path[i].strip()
    sub_paths = full_path.split('/')
    full_path_prefix = full_path[:-len(sub_paths[-1])]
    if not os.path.exists(write_path_prefix + full_path_prefix):
        os.makedirs(write_path_prefix + full_path_prefix)
    print('%d\t%s' % (i, sub_paths[-1]))

    img = cv2.imread(full_path)
    aligned_img, new_land = align_face_49pts(img, all_imgs_land[i], box_enlarge, img_size)
    cv2.imwrite(write_path_prefix + full_path, aligned_img)
    all_imgs_new_land[i, :] = new_land

np.savetxt(list_path_prefix+'BP4D_combine_1_2_land.txt', all_imgs_new_land, fmt='%f', delimiter='\t')