import torch


x = torch.tensor([1,2,3])
print(x.shape)
y = torch.tensor([4,5,6])

#addition 
z = x + y   #element wise addition
print(z)


#subtraction 
z= x - y   #element wise subtraction

#division
z = torch.true_divide(x,y)
print(z.dtype)

#multiplication
z = x * y
print(z)

#inplace operations
t = torch.ones(3)
t.add_(x)  #add x to t and update t
print(t)


#exponentiation
z= x**2
print(z)


#simple comparision
z= x > 0
print(z)


#Matrix multiplication
x1 = torch.rand((3,5))
x2 = torch.rand((5,2))
x3 = torch.mm(x1,x2) #3x2

#matrix exponentiation
x4 = torch.rand((5,5))   # ---> must be a square matrix 
m_exp = x4.matrix_power(2)
print(m_exp)

#dot product
z = torch.dot(x,y)
print(z)

#batch matrix multiplication
batch = 32
m = 10
n = 20
p = 30

tensor1 = torch.rand((batch , n , m))
tensor2 = torch.rand((batch , m , p))
out_bmm = torch.bmm(tensor1 , tensor2)


#braodcasting example
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2
z = x1 ** x2

#useful tensor operations
x = torch.tensor([1,2,3])
sum_x = torch.sum(x , dim=0)   #dim=0 represents row
x = torch.tensor([[1,2,3] , [4, 5, 6]])
sum_x = torch.sum(x, dim =0)  #tensor([5, 7, 9])
sum_x = torch.sum(x, dim =1) #tensor([ 6, 15])

x = torch.tensor([1,2,3])
values , indices = torch.max(x, dim =0)
values , indices = torch.min(x, dim =0)
print(values , indices)

#finding absolute values
abs_x = torch.abs(x)
print(abs_x)

#argmax
z = torch.argmax(x , dim =0)   #gives the indices
print(z)

#argmin
z = torch.argmin(x , dim =0)
print(z)

mean_x = torch.mean(x.float() , dim =0) #for mean we have to convert it into float

#if tensors are equal
z = torch.eq(x,y)

#to sort the values
sorted_y , indices = torch.sort(y , dim=0 , descending=False)  #sort in ascending order row wise

#clamp the values less than 0 to 0
z = torch.clamp(x , min=0)

#see if any value is True
x = torch.tensor([1,0,1,1,1,1])
z = torch.any(x) #see if any element in x is True or 1 ; if yes will return True
z = torch.all(x) #see if all element in x is 1 ; if yes will return True
print(z)