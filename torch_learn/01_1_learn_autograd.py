from __future__ import print_function
import torch

# x = torch.ones(2, 2, requires_grad=True)
x = torch.tensor([[2,2],
                  [2,2]], dtype=torch.float16, requires_grad=True)
print(x)
y = x**2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()

print(z, out)

out.backward()
print(x.grad) # d(out)/dx
# print(y.grad) # d(out)/dy

'''
out = 1/4 * sum(z)
    = 1/4 * sum(3 * y^2)
    = 3/4 * sum(y^2)
    = 3/4 * sum(x^4)
    = 3/4 * (x1^4 + x2^4 + x3^4 + x4^4)

d(out)/dx1 = 3/4 * 4 * x1^3 = 3/4 * 4 * 2^3 = 3/4 * 4 * 8 = 3 * 8 = 24
d(out)/dx2 = 3/4 * 4 * x2^3 = 3/4 * 4 * 2^3 = 3/4 * 4 * 8 = 3 * 8 = 24
d(out)/dx3 = 3/4 * 4 * x3^3 = 3/4 * 4 * 2^3 = 3/4 * 4 * 8 = 3 * 8 = 24
d(out)/dx4 = 3/4 * 4 * x4^3 = 3/4 * 4 * 2^3 = 3/4 * 4 * 8 = 3 * 8 = 24
'''

out2 = x.sum()
out2.backward()
print(x.grad) # d(out2)/dx

'''
out2 = x1 + x2 + x3 + x4
usually, d(out2)/dx = [[1, 1]
                        [1, 1]]

But backward() will accumulate the gradient in x.grad instead of overwriting it.
So, the output will be [[25, 25]
                        [25, 25]]
'''
