from typing import Sequence
import torch

x = torch.arange(9)

x_3x3 = x.view(3,3) 
x_3x3 = x.reshape(3,3)  #both these reshapes x from 9 elements to 3,3 dimension

y = x_3x3.t()
print(y.shape)

x_3 = y.reshape(9)  #convert back to 9 element dimension
print(x_3.shape)

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2) , dim=0).shape)  #concatenate across row dimension
print(torch.cat((x1,x2) , dim=1).shape)  #concatenate across column dimension

batch = 64
x = torch.rand((batch , 2, 5))
z = x.view(batch, -1)  #reshape into (batch , 2*5)
print(z.shape)

z = x.permute(0 , 2 , 1)  #changing dimensions  ---> 0 means batch remains unchanged , 2 means 2nd dimension kept in 1st , 1 means 1st dimension kept in 2nd
print(z.shape)

x = torch.arange(10) #[10]
z = x.unsqueeze(0)  #to convert it into [1,10]   ----> imp
print(z.shape)

z = x.unsqueeze(1) #to convert it into [10,1]
print(z.shape)


x = torch.arange(10).unsqueeze(0).unsqueeze(1)  #1x1x10
print(x.shape)

z = x.squeeze(1) #it will convert 1x1x10 to 1x10    ---> imp
print(z.shape)