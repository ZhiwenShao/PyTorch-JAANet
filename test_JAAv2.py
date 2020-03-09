import argparse
import os
import torch.optim as optim
import torch.utils.data as util_data

import network
import pre_process as prep
from util import *
from data_list import ImageList

optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}


def main(config):
    ## set loss criterion
    use_gpu = torch.cuda.is_available()

    ## prepare data
    dsets = {}
    dset_loaders = {}
    dsets['test'] = ImageList(crop_size=config.crop_size, path=config.test_path_prefix, phase='test',
                                       transform=prep.image_test(crop_size=config.crop_size),
                                       target_transform=prep.land_transform(img_size=config.crop_size,
                                                                            flip_reflect=np.loadtxt(
                                                                                config.flip_reflect))
                                       )

    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    ## set network modules
    region_learning = network.network_dict[config.region_learning](input_dim=3, unit_dim=config.unit_dim)
    align_net = network.network_dict[config.align_net](crop_size=config.crop_size, map_size=config.map_size,
                                                           au_num=config.au_num, land_num=config.land_num,
                                                           input_dim=config.unit_dim * 8)
    local_attention_refine = network.network_dict[config.local_attention_refine](au_num=config.au_num,
                                                                                     unit_dim=config.unit_dim)
    local_au_net = network.network_dict[config.local_au_net](au_num=config.au_num, input_dim=config.unit_dim * 8,
                                                                 unit_dim=config.unit_dim)
    global_au_feat = network.network_dict[config.global_au_feat](input_dim=config.unit_dim * 8,
                                                                     unit_dim=config.unit_dim)
    au_net = network.network_dict[config.au_net](au_num=config.au_num, input_dim=12000, unit_dim=config.unit_dim)

    if use_gpu:
        region_learning = region_learning.cuda()
        align_net = align_net.cuda()
        local_attention_refine = local_attention_refine.cuda()
        local_au_net = local_au_net.cuda()
        global_au_feat = global_au_feat.cuda()
        au_net = au_net.cuda()

    if not os.path.exists(config.write_path_prefix + config.run_name):
        os.makedirs(config.write_path_prefix + config.run_name)
    if not os.path.exists(config.write_res_prefix + config.run_name):
        os.makedirs(config.write_res_prefix + config.run_name)

    if config.start_epoch <= 0:
        raise (RuntimeError('start_epoch should be larger than 0\n'))

    res_file = open(
        config.write_res_prefix + config.run_name + '/' + config.prefix + 'offline_AU_pred_' + str(config.start_epoch) + '.txt', 'w')
    region_learning.train(False)
    align_net.train(False)
    local_attention_refine.train(False)
    local_au_net.train(False)
    global_au_feat.train(False)
    au_net.train(False)

    for epoch in range(config.start_epoch, config.n_epochs + 1):
        region_learning.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/region_learning_' + str(epoch) + '.pth'))
        align_net.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/align_net_' + str(epoch) + '.pth'))
        local_attention_refine.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/local_attention_refine_' + str(epoch) + '.pth'))
        local_au_net.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/local_au_net_' + str(epoch) + '.pth'))
        global_au_feat.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/global_au_feat_' + str(epoch) + '.pth'))
        au_net.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/au_net_' + str(epoch) + '.pth'))

        if config.pred_AU:
            local_f1score_arr, local_acc_arr, f1score_arr, acc_arr, mean_error, failure_rate = AU_detection_evalv2(
                dset_loaders['test'], region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=use_gpu)
            print('epoch =%d, local f1 score mean=%f, local accuracy mean=%f, '
                  'f1 score mean=%f, accuracy mean=%f, mean error=%f, failure rate=%f' % (
                  epoch, local_f1score_arr.mean(),
                  local_acc_arr.mean(), f1score_arr.mean(),
                  acc_arr.mean(), mean_error, failure_rate))
            print('%d\t%f\t%f\t%f\t%f\t%f\t%f' % (epoch, local_f1score_arr.mean(),
                                                  local_acc_arr.mean(), f1score_arr.mean(),
                                                  acc_arr.mean(), mean_error, failure_rate), file=res_file)
        if config.vis_attention:
            if not os.path.exists(config.write_res_prefix + config.run_name + '/vis_map/' + str(epoch)):
                os.makedirs(config.write_res_prefix + config.run_name + '/vis_map/' + str(epoch))
            if not os.path.exists(config.write_res_prefix + config.run_name + '/overlay_vis_map/' + str(epoch)):
                os.makedirs(config.write_res_prefix + config.run_name + '/overlay_vis_map/' + str(epoch))

            vis_attention(dset_loaders['test'], region_learning, align_net, local_attention_refine,
                              config.write_res_prefix, config.run_name, epoch, use_gpu=use_gpu)

    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=44, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=12, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=12, help='number of total epochs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pred_AU', type=str2bool, default=True)
    parser.add_argument('--vis_attention', type=str2bool, default=False)

    parser.add_argument('--region_learning', type=str, default='HMRegionLearning')
    parser.add_argument('--align_net', type=str, default='AlignNet')
    parser.add_argument('--local_attention_refine', type=str, default='LocalAttentionRefine')
    parser.add_argument('--local_au_net', type=str, default='LocalAUNetv2')
    parser.add_argument('--global_au_feat', type=str, default='HLFeatExtractor')
    parser.add_argument('--au_net', type=str, default='AUNet')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--run_name', type=str, default='/JAAv2')
    parser.add_argument('--prefix', type=str, default='')

    # Directories.
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--test_path_prefix', type=str, default='data/list/BP4D_part3')
    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)