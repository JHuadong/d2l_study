import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print('--------------------------------------Question 2.5.1--------------------------------------------')
    # The reasons are below:
    # 1. Computation graph construction: When computing first-order derivatives, PyTorch only needs to construct
    # the computation graph once and then perform backpropagation as needed. However, when computing second-order
    # derivatives, the computation graph must be constructed twice: once to compute the first-order derivative and
    # again to compute the derivative of the first-order derivative—that is, the second-order derivative. Therefore,
    # computing second-order derivatives requires more computation graph construction operations.

    # 2. Memory Usage: When computing first-order derivatives, PyTorch only needs to retain the gradient values
    # of the first-order derivatives, whereas computing second-order derivatives requires retaining the gradient
    # values of both the first-order and second-order derivatives, which consumes more memory.

    # 3. Increased Computational Load: Computing second-order derivatives requires additional computations on the
    # first-order derivatives, which increases both the computational load and the computation time.

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.5.2--------------------------------------------')
    # If you run the backward function and then immediately run it again, an error will occur with the message:
    # “Attempting a second backward (or directly accessing saved tensors after they have been released).”
    # Calling .backward() or autograd.grad() releases the intermediate values stored in the graph. If you need
    # to perform a second backward, or if you need to access the saved tensors after the backward call,
    # set retain_graph=True.

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.5.3--------------------------------------------')
    def f(a):
        b = a * 2
        while b.norm() > 1000:
            b = b * 2
        if b.sum() > 0:
            c = b
        else:
            c = 100 * b
        return c
    a = torch.randn(size=(1, 3), requires_grad=True)
    d = f(a)
    try:
        d.backward(torch.ones_like(d))
        print(a.grad)
    except Exception as e:
        print(e)
    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.5.5--------------------------------------------')


    def simple_sin(x):
        return torch.sin(x)

    def df(x):
        x.requires_grad_()
        y = simple_sin(x)
        y.backward(torch.ones_like(y))
        return x.grad

    x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 200)

    y1 = simple_sin(x)
    y2 = df(x)

    plt.plot(x.detach().numpy(), y1.detach().numpy(), label='f(x)=sin(x)')
    plt.plot(x.detach().numpy(), y2.detach().numpy(), label='f(x)=df(x)/dx')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print('------------------------------------------------------------------------------------------------\n\n')

