def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001):
    lr = init_lr * (1 + gamma * iter_num) ** (-power)

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1

    return optimizer


def step_lr_scheduler(param_lr, optimizer, iter_num, gamma, stepsize, init_lr=0.001):
    '''Decay learning rate by a factor of gamma every stepsize epochs.'''
    lr = init_lr * (gamma ** (iter_num // stepsize))

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1

    return optimizer


def lambda_lr_scheduler(param_lr, optimizer, epoch, n_epochs, offset, decay_start_epoch, init_lr=0.001):
    lr = init_lr * (1.0 - max(0, epoch + offset - decay_start_epoch) / float(n_epochs - decay_start_epoch + 1))

    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        i += 1

    return optimizer


schedule_dict = {'inv': inv_lr_scheduler, 'step': step_lr_scheduler, 'lambda': lambda_lr_scheduler}
