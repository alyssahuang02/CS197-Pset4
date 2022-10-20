import torch

x = torch.rand(2, 3)
y = torch.rand(2, 3, requires_grad=True)

print(x.requires_grad)
print(y.requires_grad)

z = x + y
print(z.requires_grad)