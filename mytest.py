import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class TestConv2d(nn.Module):
    """
    Module for Conv2d testing
    """

    def __init__(self, inp=10, out=16, kernel_size=3):
        super(TestConv2d, self).__init__()
        self.conv2d = nn.Conv2d(inp, out, stride=1, kernel_size=kernel_size, bias=True)

    def forward(self, x):
        x = self.conv2d(x)
        return x

model = TestConv2d()
input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))
from pytorch2keras.converter import pytorch_to_keras
# we should specify shape of the input tensor
k_model = pytorch_to_keras(model, input_var, [(10, None, None,)], verbose=True)