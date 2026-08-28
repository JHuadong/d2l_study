import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print('--------------------------------------Question 2.4.1--------------------------------------------')
    # define function
    def f(x):
        return x**3 - 1/x

    # define tangent of x = point
    def f_tangent(f, x, point):
        h = 1e-4
        grad = (f(point + h) - f(point)) / h
        return grad*(x - point) + f(point)

    # plot picture of function and tangent
    x = np.arange(0.1, 2.0, 0.01)
    y = f(x)
    y_tangent = f_tangent(f, x=x, point=1)
    plt.plot(x, y, label='f(x)')
    plt.plot(x, y_tangent, label='Tangent line at x=1')
    plt.legend()
    plt.title('Graph of f(x) and its tangent line at x=1')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.4.2--------------------------------------------')

    x = torch.arange(2.0, requires_grad=True)
    y = 3*x[0]**2 + 5*torch.exp(x[1])
    y.backward()

    print('The gradient of f(x) is', x.grad)
    print('Verify the Gradient: ', x.grad == torch.tensor([6 * x[0], 6 * torch.exp(x[1])]))

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.4.3--------------------------------------------')

    x = torch.arange(4.0, requires_grad=True)
    y = torch.dot(x, x).sqrt()
    y.backward()
    print('The gradient of f(x) is', x.grad)
    print('Verify the result: ', x.grad == x/(torch.dot(x, x).sqrt()))

    print('------------------------------------------------------------------------------------------------\n\n')