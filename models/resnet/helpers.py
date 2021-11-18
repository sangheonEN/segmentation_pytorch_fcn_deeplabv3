import torch

def prepare_optim(opts, model):

    # optimizer Adam
    optim = torch.optim.Adam(model.parameters(),
                            lr=opts.cfg['lr'],
                            weight_decay=opts.cfg['weight_decay'])

    if opts.resume:
        checkpoint = torch.load(opts.resume)
        optim.load_state_dict(checkpoint['optim_state_dict'])

    return optim
