import torch
import os
from glob import glob

def save_network(net, label, which_iter, opt, latest=False):
    if latest:
        save_path = os.path.join(opt.checkpoint_path, f'{label}_latest_net_{which_iter}.pth')
        old_paths = glob(os.path.join(opt.checkpoint_path, f"{label}_latest_net_*.pth"))
    else:
        save_path = os.path.join(opt.checkpoint_path, f'{label}_net_{which_iter}.pth')

    torch.save(net.state_dict(), save_path)

    if latest:
        for old_path in old_paths:
            open(old_path, 'w').close()
            os.unlink(old_path)

def load_state_dict(net, state_dict, strict=True):
    if not strict:
        # remove the keys which don't match in size
        model_dict = net.state_dict()
        pop_list = []
        for key in state_dict:
            if key in model_dict:
                if model_dict[key].shape != state_dict[key].shape:
                    pop_list.append(key)
                    print(f"Size mismatch for {key}")
            else:
                pop_list.append(key)
                print(f"Key missing in checkpoint for {key}")
        for key in pop_list:
            state_dict.pop(key)
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)
    else:
        net.load_state_dict(state_dict)

def load_network(net, label, opt, override_iter=None, override_load_path=None):
    which_iter = override_iter if override_iter is not None else opt.which_iter
    load_path = override_load_path if override_load_path is not None else opt.load_path
    if load_path is not None and which_iter > 0:
        latest_load_path = os.path.join(load_path, f'{label}_latest_net_{which_iter}.pth')
        load_path = os.path.join(load_path, f'{label}_net_{which_iter}.pth')
        if not os.path.exists(load_path):
            if not os.path.exists(latest_load_path):
                raise ValueError(f"No checkpoint for {label} net at iter {which_iter} and path {load_path}")
            else:
                load_path = latest_load_path
        load_state_dict(net, torch.load(load_path), strict=not opt.not_strict)
        print(f"Loading checkpoint for {label} net from {load_path}")

    elif opt.cont_train and which_iter > 0:
        load_paths = glob(os.path.join(opt.save_path, "checkpoints", f"*-{opt.name}", f"{label}_latest_net_{which_iter}.pth"))
        load_paths += glob(os.path.join(opt.save_path, "checkpoints", f"*-{opt.name}", f"{label}_net_{which_iter}.pth"))
        assert len(load_paths) > 0, f"Did not find any checkpoint for {label} net at iter {which_iter} and name {opt.name}"
        assert len(load_paths) == 1, f"Too many checkpoint candidates for {label} net at iter {which_iter} and name {opt.name}:\n{load_paths}"
        load_path = load_paths[0]
        load_state_dict(net, torch.load(load_path), strict=not opt.not_strict)
        print(f"Loading checkpoint for {label} net from {load_path}")

    else:
        print(f"Loading untrained {label} net")

    return net

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)