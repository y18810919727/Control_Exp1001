
# -*- coding:utf8 -*-
import torch.nn as nn

# 用于网络复制
def soft_update(net, target_net, soft_tau=1e-2):
    if not issubclass(net.__class__, nn.Module):
        raise TypeError('Net is not an instance of any subclass of torch.nn.Model')
    if not issubclass(target_net.__class__, nn.Module):
        raise TypeError('Target_net is not  an instance of any subclass of torch.nn.Model')
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data*(1.0 - soft_tau) + param.data * soft_tau
        )


def copy_net(net, target_net):
    soft_update(net, target_net,soft_tau=1.0)