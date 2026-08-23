import torch

if __name__ == '__main__':

    print('--------------------------------------Question 2.3.1--------------------------------------------')
    A = torch.arange(20).reshape(5, 4)
    print(A.T.T == A)

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.3.2--------------------------------------------')
    A = torch.rand(4, 4)
    B = torch.rand(4, 4)
    print(A.T + B.T == (A + B).T)

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.3.4--------------------------------------------')
    X = torch.arange(24).reshape(2, 3, 4)
    print(len(X))

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.3.5--------------------------------------------')
    n = int(torch.rand(1) * 10) + 1
    m = int(torch.rand(1) * 10) + 1
    X = torch.rand(n, m)
    print(torch.tensor([len(X), len(X)]) == torch.tensor(X.shape))

    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.3.6--------------------------------------------')
    A = torch.arange(20).reshape(5, 4)
    try:
        print(A / A.sum(dim=1))
    except Exception as e:
        print(e)
    # The code will throw an error. The result of `A.sum(axis=1)` has one fewer axis than `A` and the length of axis 1
    # does not match that of `A`, so it cannot be broadcast. You can try adding `keepdims=True` to achieve the desired
    # result.
    A = torch.arange(20).reshape(5, 4)
    print(A / A.sum(dim=1, keepdim=True))
    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.3.7--------------------------------------------')
    A = torch.arange(24).reshape(2, 3, 4)
    print(
        f'在轴0上的求和：{A.sum(dim=0).shape}\n在轴1上的求和：{A.sum(dim=1).shape}\n在轴2上的求和：{A.sum(dim=2).shape}')
    print('------------------------------------------------------------------------------------------------\n\n')

    print('--------------------------------------Question 2.3.7--------------------------------------------')
    x = torch.randn(2, 3, 4)
    # Compute the norm in different dimensions
    norm_dim0 = torch.linalg.norm(x, dim=0)
    print(norm_dim0.shape)

    norm_dim1 = torch.linalg.norm(x, dim=1)
    print(norm_dim1.shape)

    norm_dim2 = torch.linalg.norm(x, dim=2)
    print(norm_dim2.shape)
    # answer:  If you sum along a particular axis, that axis disappears.
    print('------------------------------------------------------------------------------------------------\n\n')