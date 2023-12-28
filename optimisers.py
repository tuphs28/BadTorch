import torch
import torch.nn as nn
import torch.nn.functional as F


class Optimiser:
    """
    Parent class for optimisers

    Args:
        parameters (list): list of model parameters (tensors) to optimiser
        lr (float): learning rate
        grad_clip (int): if non-zero integer provided, perform gradient clipping with this value
    """

    def __init__(self, parameters: list, lr: float = 1e-3, grad_clip: int = 0) -> None:
        self.parameters = parameters
        self.lr = lr
        self.grad_clip = grad_clip

    def zero_grad(self) -> None:
        """
        Set all gradients to None to avoid accumulation
        """
        for parameter in self.parameters:
            parameter.grad = None

    def clip_gradient(self, x: torch.Tensor) -> None:
        """
        Clip gradients so values do no exceed provided grad_clip attribute

        Args:
            x (torch.Tensor): parameter tensor whose gradient is to be clipped
        """
        clipped_grad = torch.clip(x.grad, -self.grad_clip, self.grad_clip)
        x.grad = clipped_grad

class SGD(Optimiser):
    """
    Basic implementation of stochastic gradient descent
    """

    def step(self) -> None:
        """
        Perform step of SGD optimisation
        """
        for parameter in self.parameters:
            
            # perform gradient clipping if grad_clip value provided
            if self.grad_clip:
                self.clip_gradient(parameter)

            # perform step of optimisation
            parameter.data -= self.lr * parameter.grad


class Adam(Optimiser):
    """
    Implementation of Adam optimisation algorithm

    Args:
        parameters (list): list of model parameters (tensors) to optimiser
        lr (float): learning rate
        beta1 (float): parameter to control running average estimate of first moment
        beta2 (float): parameter to control running average estimate of second moment
        eps (float): parameter to avoid dividing by zero when correcting bias
        grad_clip (int): if non-zero integer provided, perform gradient clipping with this value   
        device (torch.device): device to place buffers on, must be same as parameters that are being optimised   
    """

    def __init__(self, parameters: list, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, grad_clip: int = 0, device: torch.device = torch.device("cpu")) -> None:
        super().__init__(parameters, lr, grad_clip)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # step count variable for use in bias correction
        self.t = 0

        self.m1s = []
        self.m2s = []
        for parameter in self.parameters:
            self.m1s.append(torch.ones(parameter.shape).to(device))
            self.m2s.append(torch.ones(parameter.shape).to(device))

    def step(self) -> None:
        """
        Perform step of Adam optimisation
        """
        # update step count for bias correction
        self.t += 1

        for parameter_idx, parameter in enumerate(self.parameters):

            # perform gradient clipping if grad_clip value provided
            if self.grad_clip:
                self.clip_gradient(parameter)

            # update running average estimates of first and second moments
            self.m1s[parameter_idx] = self.beta1 * self.m1s[parameter_idx] + (1-self.beta1) * parameter.grad
            self.m2s[parameter_idx] = self.beta2 * self.m2s[parameter_idx] + (1-self.beta2) * parameter.grad ** 2

            # perform bias correction on first and second moments
            m1_hat = self.m1s[parameter_idx] / (1 - self.beta1 ** self.t)
            m2_hat = self.m2s[parameter_idx] / (1 - self.beta2 ** self.t)

            parameter.data -= self.lr * m1_hat / (torch.sqrt(m2_hat) + self.eps)
