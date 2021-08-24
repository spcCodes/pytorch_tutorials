import torch

batch_size = 10
features = 25
x = torch.rand(batch_size , features)

print(x[0].shape)   #x[0 , :] --> 0th row and all columns

print(x[:,0].shape) #all rows and 0th column

print(x[2,0:10])  #2nd row and 10 column values out of 25

x[0,0] = 100 #assigning values of 0,0 element to 100

#fancy indeximg
x = torch.arange(10)
print(x)
indices = [2,5,7] 
print(x[indices]) # --> tensor([2, 5, 7])

x = torch.rand((3,5))
rows=torch.tensor([1,0])
columns= torch.tensor([4,0])
print(x[rows,columns])  #will print two elements x[1,4] and x[0,0]

#advanced indexing
x= torch.arange(10)
print(x[(x<2) | (x>8)])
print(x[(x<2) & (x>8)])
print(x[x.remainder(2)==0])  #where x%2 ==0 those values

#useful operations
print(torch.where(x>5 , x , x*2)) #look at x --> where x > 5 keep x else put x*2
print(torch.tensor([0,0,1,1,1,2,4,5,5,5]).unique())
print(x.ndimension()) #number of dimension of x , in this case it is 1
print(x.numel())  #number of elements in x

