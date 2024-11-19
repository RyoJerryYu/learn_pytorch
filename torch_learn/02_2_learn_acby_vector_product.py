from __future__ import print_function
import torch

x = torch.tensor([1, 1, 1], dtype=torch.float16, requires_grad=True)
print(x)  # tensor([1., 1., 1.], dtype=torch.float16, requires_grad=True)
print(x.data.norm())  # tensor(1.7324, dtype=torch.float16)

y = x
i = 0
while y.data.norm() < 1000:
    y = y * 2
    i = i + 1
print(y)  # tensor([1024., 1024., 1024.], dtype=torch.float16, grad_fn=<MulBackward0>)
print(i)  # 10

'''
The Jacobian matrix of y = x * 2^10 will be:
[[dy1/dx1, dy1/dx2, dy1/dx3],
 [dy2/dx1, dy2/dx2, dy2/dx3],
 [dy3/dx1, dy3/dx2, dy3/dx3]]
 
 = [[2^10, 0, 0],
    [0, 2^10, 0],
    [0, 0, 2^10]]

But torch will not compute the full Jacobian matrix, only the vector-Jacobian product.

If v = [v1, v2, v3], then the vector-Jacobian product will be:
[[v1, v2, v3]] * [[2^10, 0, 0],
                  [0, 2^10, 0],
                  [0, 0, 2^10]]
                = [v1 * 2^10, v2 * 2^10, v3 * 2^10]
'''

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float16)
y.backward(v)
print(x.grad)  # tensor([1.0240, 10.2400, 0.0010], dtype=torch.float16)
