import torch

#initialise tensor
device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([ [1,2,3] , [4,5,6] ] , dtype = torch.float32 , device=device , requires_grad=True)

# print(my_tensor)

#other common initialisation methods
x= torch.empty(size=(3,3))
x = torch.zeros(size=(3,3))
x = torch.rand((3,3)) #uniform distribution between 0 and 1
x= torch.ones((3,3))
x= torch.eye(5,5)
x = torch.arange(start=1 , end=10 , step=2)
x = torch.linspace(start=0.1 , end=1 , steps=20)
x=torch.empty((1,5)).normal_(mean=0 , std=1)
x=torch.empty((1,5)).uniform_(0,1)
x=torch.diag(torch.ones(5))



#convert tensors to other types
x = torch.arange(4)
print(x.bool()) #boolean true/false   ---> imp
print(x.short())  #int16
print(x.long())  #int64
print(x.half()) #float16 ---> #used in newer version of gpus
print(x.float()) #float32 ---> imp
print(x.double())  #float64  ---> imp



#arrays to tensors conversion
import numpy as np
np_array = np.zeros((5,5))
tensor_array = torch.from_numpy(np_array) #  ---> to tensors
np_array_back = tensor_array.numpy()   # ---> to numpy array


