#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json
import torch

from torch.autograd import Function



class LinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)
        return output

    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0).squeeze(0)

        #return grad_input, grad_weight, grad_bias
        return grad_input, None, None

from torch.autograd import gradcheck
linear = LinearFunction.apply


input = torch.randn(20, 20, dtype=torch.double, requires_grad=True), \
        torch.randn(30, 20, dtype=torch.double, requires_grad=True), \
        torch.randn(30, dtype=torch.double, requires_grad=False)
input[0].requires_grad=True
x = linear(*input)
x.sum().backward()
print(input[0].grad)

# print(gradcheck(linear, input))
#print(type(input))




