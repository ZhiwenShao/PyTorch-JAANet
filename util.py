import torch
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import math

def str2bool(v):
    return v.lower() in ('true')

def tensor2img(img):
    img = img.data.numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = (np.transpose(img, (1, 2, 0))+ 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def save_img(img, name, path):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(path + name + '.png')
    return img


def AU_detection_evalv1(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=True, fail_threshold=0.1):
    missing_label = 9
    for i, batch in enumerate(loader):
        input, land, biocular, au = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat = local_au_net(region_feat, output_aus_map)
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
        aus_output = au_net(concat_au_feat)
        aus_output = (aus_output[:,1,:]).exp()

        if i == 0:
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
            all_pred_land = align_output.data.cpu().float()
            all_land = land.data.cpu().float()
        else:
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)
            all_pred_land = torch.cat((all_pred_land, align_output.data.cpu().float()), 0)
            all_land = torch.cat((all_land, land.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()
    pred_land = all_pred_land.data.numpy()
    GT_land = all_land.data.numpy()

    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    # landmarks
    errors = np.zeros((GT_land.shape[0], int(GT_land.shape[1] / 2)))
    mean_errors = np.zeros(GT_land.shape[0])
    for i in range(GT_land.shape[0]):
        left_eye_x = GT_land[i, (20 - 1) * 2:(26 - 1) * 2:2]
        l_ocular_x = left_eye_x.mean()
        left_eye_y = GT_land[i, (20 - 1) * 2 + 1:(26 - 1) * 2 + 1:2]
        l_ocular_y = left_eye_y.mean()

        right_eye_x = GT_land[i, (26 - 1) * 2:(32 - 1) * 2:2]
        r_ocular_x = right_eye_x.mean()
        right_eye_y = GT_land[i, (26 - 1) * 2 + 1:(32 - 1) * 2 + 1:2]
        r_ocular_y = right_eye_y.mean()

        biocular = math.sqrt((l_ocular_x - r_ocular_x) ** 2 + (l_ocular_y - r_ocular_y) ** 2)

        for j in range(0, GT_land.shape[1], 2):
            errors[i, int(j / 2)] = math.sqrt((GT_land[i, j] - pred_land[i, j]) ** 2 + (
                    GT_land[i, j + 1] - pred_land[i, j + 1]) ** 2) / biocular

        mean_errors[i] = errors[i].mean()
    mean_error = mean_errors.mean()

    failure_ind = np.zeros(len(GT_land))
    failure_ind[mean_errors > fail_threshold] = 1
    failure_rate = failure_ind.sum() / failure_ind.shape[0]

    return f1score_arr, acc_arr, mean_error, failure_rate


def AU_detection_evalv2(loader, region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=True, fail_threshold = 0.1):
    missing_label = 9
    for i, batch in enumerate(loader):
        input, land, biocular, au  = batch
        if use_gpu:
            input, land, au = input.cuda(), land.cuda(), au.cuda()

        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        local_au_out_feat, local_aus_output = local_au_net(region_feat, output_aus_map)
        local_aus_output = (local_aus_output[:, 1, :]).exp()
        global_au_out_feat = global_au_feat(region_feat)
        concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat.detach()), 1)
        aus_output = au_net(concat_au_feat)
        aus_output = (aus_output[:,1,:]).exp()

        if i == 0:
            all_local_output = local_aus_output.data.cpu().float()
            all_output = aus_output.data.cpu().float()
            all_au = au.data.cpu().float()
            all_pred_land = align_output.data.cpu().float()
            all_land = land.data.cpu().float()
        else:
            all_local_output = torch.cat((all_local_output, local_aus_output.data.cpu().float()), 0)
            all_output = torch.cat((all_output, aus_output.data.cpu().float()), 0)
            all_au = torch.cat((all_au, au.data.cpu().float()), 0)
            all_pred_land = torch.cat((all_pred_land, align_output.data.cpu().float()), 0)
            all_land = torch.cat((all_land, land.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    local_AUoccur_pred_prob = all_local_output.data.numpy()
    AUoccur_actual = all_au.data.numpy()
    pred_land = all_pred_land.data.numpy()
    GT_land = all_land.data.numpy()
    # np.savetxt('BP4D_part1_pred_land_49.txt', pred_land, fmt='%.4f', delimiter='\t')
    np.savetxt('B3D_val_predAUprob-2_all_.txt', AUoccur_pred_prob, fmt='%f',
               delimiter='\t')
    # AUs
    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1
    local_AUoccur_pred = np.zeros(local_AUoccur_pred_prob.shape)
    local_AUoccur_pred[local_AUoccur_pred_prob < 0.5] = 0
    local_AUoccur_pred[local_AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))
    local_AUoccur_pred = local_AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    local_f1score_arr = np.zeros(AUoccur_actual.shape[0])
    local_acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]
        local_curr_pred = local_AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]
        local_new_curr_pred = local_curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)
        local_f1score_arr[i] = f1_score(new_curr_actual, local_new_curr_pred)
        local_acc_arr[i] = accuracy_score(new_curr_actual, local_new_curr_pred)

    # landmarks
    errors = np.zeros((GT_land.shape[0], int(GT_land.shape[1] / 2)))
    mean_errors = np.zeros(GT_land.shape[0])
    for i in range(GT_land.shape[0]):
        left_eye_x = GT_land[i, (20 - 1) * 2:(26 - 1) * 2:2]
        l_ocular_x = left_eye_x.mean()
        left_eye_y = GT_land[i, (20 - 1) * 2 + 1:(26 - 1) * 2 + 1:2]
        l_ocular_y = left_eye_y.mean()

        right_eye_x = GT_land[i, (26 - 1) * 2:(32 - 1) * 2:2]
        r_ocular_x = right_eye_x.mean()
        right_eye_y = GT_land[i, (26 - 1) * 2 + 1:(32 - 1) * 2 + 1:2]
        r_ocular_y = right_eye_y.mean()

        biocular = math.sqrt((l_ocular_x - r_ocular_x) ** 2 + (l_ocular_y - r_ocular_y) ** 2)

        for j in range(0, GT_land.shape[1], 2):
            errors[i, int(j / 2)] = math.sqrt((GT_land[i, j] - pred_land[i, j]) ** 2 + (
                    GT_land[i, j + 1] - pred_land[i, j + 1]) ** 2) / biocular

        mean_errors[i] = errors[i].mean()
    mean_error = mean_errors.mean()

    failure_ind = np.zeros(len(GT_land))
    failure_ind[mean_errors > fail_threshold] = 1
    failure_rate = failure_ind.sum() / failure_ind.shape[0]

    return local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate


def vis_attention(loader, region_learning, align_net, local_attention_refine, write_path_prefix, net_name, epoch, alpha = 0.5, use_gpu=True):
    for i, batch in enumerate(loader):
        input, land, biocular, au = batch
        if i > 1:
            break
        if use_gpu:
            input = input.cuda()
        region_feat = region_learning(input)
        align_feat, align_output, aus_map = align_net(region_feat)
        if use_gpu:
            aus_map = aus_map.cuda()
        output_aus_map = local_attention_refine(aus_map.detach())
        # spatial_attention = torch.sum(region_feat, 1, True)
        spatial_attention = output_aus_map
        if i == 0:
            all_input = input.data.cpu().float()
            all_spatial_attention = spatial_attention.data.cpu().float()
        else:
            all_input = torch.cat((all_input, input.data.cpu().float()), 0)
            all_spatial_attention = torch.cat((all_spatial_attention, spatial_attention.data.cpu().float()), 0)

    for i in range(all_spatial_attention.shape[0]):
        background = save_img(all_input[i], 'input', write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_')
        for j in range(all_spatial_attention.shape[1]):
            fig, ax = plt.subplots()
            # print(all_spatial_attention[i,j].max(), all_spatial_attention[i,j].min())
            # cax = ax.imshow(all_spatial_attention[i,j], cmap='jet', interpolation='bicubic')
            cax = ax.imshow(all_spatial_attention[i, j], cmap='jet', interpolation='bicubic', vmin=0, vmax=1)
            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            #        cbar = fig.colorbar(cax)
            fig.savefig(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

        for j in range(all_spatial_attention.shape[1]):
            overlay = Image.open(write_path_prefix + net_name + '/vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png')
            overlay = overlay.resize(background.size, Image.ANTIALIAS)
            background = background.convert('RGBA')
            overlay = overlay.convert('RGBA')
            new_img = Image.blend(background, overlay, alpha)
            new_img.save(write_path_prefix + net_name + '/overlay_vis_map/' + str(epoch) +
                        '/' + str(i) + '_au_' + str(j) + '.png', 'PNG')