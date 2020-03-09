import argparse
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as util_data
import itertools

import network
import pre_process as prep
import lr_schedule
from util import *
from data_list import ImageList

optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}


def main(config):
    ## set loss criterion
    use_gpu = torch.cuda.is_available()
    au_weight = torch.from_numpy(np.loadtxt(config.train_path_prefix + '_weight.txt'))
    if use_gpu:
        au_weight = au_weight.float().cuda()
    else:
        au_weight = au_weight.float()

    ## prepare data
    dsets = {}
    dset_loaders = {}

    dsets['train'] = ImageList(crop_size=config.crop_size, path=config.train_path_prefix,
                                       transform=prep.image_train(crop_size=config.crop_size),
                                       target_transform=prep.land_transform(img_size=config.crop_size,
                                                                            flip_reflect=np.loadtxt(
                                                                                config.flip_reflect)))

    dset_loaders['train'] = util_data.DataLoader(dsets['train'], batch_size=config.train_batch_size,
                                                 shuffle=True, num_workers=config.num_workers)

    dsets['test'] = ImageList(crop_size=config.crop_size, path=config.test_path_prefix, phase='test',
                                       transform=prep.image_test(crop_size=config.crop_size),
                                       target_transform=prep.land_transform(img_size=config.crop_size,
                                                                            flip_reflect=np.loadtxt(
                                                                                config.flip_reflect))
                                       )

    dset_loaders['test'] = util_data.DataLoader(dsets['test'], batch_size=config.eval_batch_size,
                                                shuffle=False, num_workers=config.num_workers)

    ## set network modules
    region_learning = network.network_dict[config.region_learning](input_dim=3, unit_dim = config.unit_dim)
    align_net = network.network_dict[config.align_net](crop_size=config.crop_size, map_size=config.map_size,
                                                           au_num=config.au_num, land_num=config.land_num, input_dim=config.unit_dim*8)
    local_attention_refine = network.network_dict[config.local_attention_refine](au_num=config.au_num, unit_dim=config.unit_dim)
    local_au_net = network.network_dict[config.local_au_net](au_num=config.au_num, input_dim=config.unit_dim*8,
                                                                                     unit_dim=config.unit_dim)
    global_au_feat = network.network_dict[config.global_au_feat](input_dim=config.unit_dim*8,
                                                                                     unit_dim=config.unit_dim)
    au_net = network.network_dict[config.au_net](au_num=config.au_num, input_dim = 12000, unit_dim = config.unit_dim)


    if config.start_epoch > 0:
        print('resuming model from epoch %d' %(config.start_epoch))
        region_learning.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/region_learning_' + str(config.start_epoch) + '.pth'))
        align_net.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/align_net_' + str(config.start_epoch) + '.pth'))
        local_attention_refine.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/local_attention_refine_' + str(config.start_epoch) + '.pth'))
        local_au_net.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/local_au_net_' + str(config.start_epoch) + '.pth'))
        global_au_feat.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/global_au_feat_' + str(config.start_epoch) + '.pth'))
        au_net.load_state_dict(torch.load(
            config.write_path_prefix + config.run_name + '/au_net_' + str(config.start_epoch) + '.pth'))

    if use_gpu:
        region_learning = region_learning.cuda()
        align_net = align_net.cuda()
        local_attention_refine = local_attention_refine.cuda()
        local_au_net = local_au_net.cuda()
        global_au_feat = global_au_feat.cuda()
        au_net = au_net.cuda()

    print(region_learning)
    print(align_net)
    print(local_attention_refine)
    print(local_au_net)
    print(global_au_feat)
    print(au_net)

    ## collect parameters
    region_learning_parameter_list = [{'params': filter(lambda p: p.requires_grad, region_learning.parameters()), 'lr': 1}]
    align_net_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, align_net.parameters()), 'lr': 1}]
    local_attention_refine_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, local_attention_refine.parameters()), 'lr': config.grad_enhance}]
    local_au_net_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, local_au_net.parameters()), 'lr': 1}]
    global_au_feat_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, global_au_feat.parameters()), 'lr': 1}]
    au_net_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, au_net.parameters()), 'lr': 1}]

    ## set optimizer
    optimizer = optim_dict[config.optimizer_type](itertools.chain(region_learning_parameter_list, align_net_parameter_list,
                                                                  local_attention_refine_parameter_list,
                                                                  local_au_net_parameter_list,
                                                                  global_au_feat_parameter_list,
                                                                  au_net_parameter_list),
                                                  lr=1.0, momentum=config.momentum, weight_decay=config.weight_decay,
                                                  nesterov=config.use_nesterov)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])

    lr_scheduler = lr_schedule.schedule_dict[config.lr_type]

    if not os.path.exists(config.write_path_prefix + config.run_name):
        os.makedirs(config.write_path_prefix + config.run_name)
    if not os.path.exists(config.write_res_prefix + config.run_name):
        os.makedirs(config.write_res_prefix + config.run_name)

    res_file = open(
        config.write_res_prefix + config.run_name + '/AU_pred_' + str(config.start_epoch) + '.txt', 'w')

    ## train
    count = 0

    for epoch in range(config.start_epoch, config.n_epochs + 1):
        if epoch > config.start_epoch:
            print('taking snapshot ...')
            torch.save(region_learning.state_dict(),
                       config.write_path_prefix + config.run_name + '/region_learning_' + str(epoch) + '.pth')
            torch.save(align_net.state_dict(),
                       config.write_path_prefix + config.run_name + '/align_net_' + str(epoch) + '.pth')
            torch.save(local_attention_refine.state_dict(),
                       config.write_path_prefix + config.run_name + '/local_attention_refine_' + str(epoch) + '.pth')
            torch.save(local_au_net.state_dict(),
                       config.write_path_prefix + config.run_name + '/local_au_net_' + str(epoch) + '.pth')
            torch.save(global_au_feat.state_dict(),
                       config.write_path_prefix + config.run_name + '/global_au_feat_' + str(epoch) + '.pth')
            torch.save(au_net.state_dict(),
                       config.write_path_prefix + config.run_name + '/au_net_' + str(epoch) + '.pth')

        # eval in the train
        if epoch > config.start_epoch:
            print('testing ...')
            region_learning.train(False)
            align_net.train(False)
            local_attention_refine.train(False)
            local_au_net.train(False)
            global_au_feat.train(False)
            au_net.train(False)

            f1score_arr, acc_arr, mean_error, failure_rate = AU_detection_evalv1(
                dset_loaders['test'], region_learning, align_net, local_attention_refine,
                local_au_net, global_au_feat, au_net, use_gpu=use_gpu)
            print('epoch =%d, f1 score mean=%f, accuracy mean=%f, mean error=%f, failure rate=%f'
                  % (epoch, f1score_arr.mean(), acc_arr.mean(), mean_error, failure_rate))
            print('%d\t%f\t%f\t%f\t%f' % (epoch, f1score_arr.mean(), acc_arr.mean(),
                                          mean_error, failure_rate), file=res_file)

            region_learning.train(True)
            align_net.train(True)
            local_attention_refine.train(True)
            local_au_net.train(True)
            global_au_feat.train(True)
            au_net.train(True)

        if epoch >= config.n_epochs:
            break

        for i, batch in enumerate(dset_loaders['train']):
            if i % config.display == 0 and count > 0:
                print('[epoch = %d][iter = %d][total_loss = %f][loss_au_softmax = %f][loss_au_dice = %f][loss_land = %f][loss_attention_refine = %f]' % (epoch, i,
                    total_loss.data.cpu().numpy(), loss_au_softmax.data.cpu().numpy(), loss_au_dice.data.cpu().numpy(), loss_land.data.cpu().numpy(), loss_attention_refine.data.cpu().numpy()))
                print('learning rate = %f %f %f %f %f %f' % (optimizer.param_groups[0]['lr'],
                                                          optimizer.param_groups[1]['lr'],
                                                          optimizer.param_groups[2]['lr'],
                                                          optimizer.param_groups[3]['lr'],
                                                          optimizer.param_groups[4]['lr'],
                                                          optimizer.param_groups[5]['lr']))
                print('the number of training iterations is %d' % (count))

            input, land, biocular, au = batch

            if use_gpu:
                input, land, biocular, au = input.cuda(), land.float().cuda(), \
                                            biocular.float().cuda(), au.long().cuda()
            else:
                au = au.long()

            optimizer = lr_scheduler(param_lr, optimizer, epoch, config.gamma, config.stepsize, config.init_lr)
            optimizer.zero_grad()

            region_feat = region_learning(input)
            align_feat, align_output, aus_map = align_net(region_feat)
            if use_gpu:
                aus_map = aus_map.cuda()
            output_aus_map = local_attention_refine(aus_map.detach())
            local_au_out_feat = local_au_net(region_feat, output_aus_map)
            global_au_out_feat = global_au_feat(region_feat)
            concat_au_feat = torch.cat((align_feat, global_au_out_feat, local_au_out_feat), 1)
            aus_output = au_net(concat_au_feat)

            loss_au_softmax = au_softmax_loss(aus_output, au, weight=au_weight)
            loss_au_dice = au_dice_loss(aus_output, au, weight=au_weight)
            loss_au = loss_au_softmax + loss_au_dice
            loss_land = landmark_loss(align_output, land, biocular)

            resize_aus_map = F.interpolate(aus_map, size=output_aus_map.size(2))
            loss_attention_refine = attention_refine_loss(output_aus_map, resize_aus_map.detach())

            total_loss = config.lambda_au * loss_au + config.lambda_land * loss_land + \
                         (config.lambda_refine/config.grad_enhance) * loss_attention_refine

            total_loss.backward()
            optimizer.step()

            count = count + 1

    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--map_size', type=int, default=44, help='size for attention maps')
    parser.add_argument('--au_num', type=int, default=12, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--train_batch_size', type=int, default=8, help='mini-batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=12, help='number of total epochs')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--region_learning', type=str, default='HMRegionLearning')
    parser.add_argument('--align_net', type=str, default='AlignNet')
    parser.add_argument('--local_attention_refine', type=str, default='LocalAttentionRefine')
    parser.add_argument('--local_au_net', type=str, default='LocalAUNetv1')
    parser.add_argument('--global_au_feat', type=str, default='HLFeatExtractor')
    parser.add_argument('--au_net', type=str, default='AUNet')
    parser.add_argument('--unit_dim', type=int, default=8, help='unit dims')
    parser.add_argument('--run_name', type=str, default='JAAv1')

    # Training configuration.
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.5, help='weight for landmark detection loss')
    parser.add_argument('--lambda_refine', type=float, default=0.0002, help='weight for AU attention refinement')
    parser.add_argument('--grad_enhance', type=float, default=2, help='weight for back-propagation enhancement')
    parser.add_argument('--display', type=int, default=100, help='iteration gaps for displaying')

    parser.add_argument('--optimizer_type', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for SGD optimizer')
    parser.add_argument('--use_nesterov', type=str2bool, default=True)
    parser.add_argument('--lr_type', type=str, default='step')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.3, help='decay factor')
    parser.add_argument('--stepsize', type=int, default=2, help='epoch for decaying lr')

    # Directories.
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--train_path_prefix', type=str, default='data/list/BP4D_combine_1_2')
    parser.add_argument('--test_path_prefix', type=str, default='data/list/BP4D_part3')

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)