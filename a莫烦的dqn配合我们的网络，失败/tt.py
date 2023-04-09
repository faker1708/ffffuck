

import torch
import torch.nn as nn



fc1 = nn.Linear(4,2)
# print(fc1)


x = torch.normal(0,1,(4,1))
print(x)


y = fc1(x)

print(y)
