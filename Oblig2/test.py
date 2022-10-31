import numpy as np


filter = np.array([[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]])

f = np.array([[2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1],
              [2,3,4,1,0,1,1,1,1,1,1,1,2,3,4,1,0,1,1,1,1,1,1,1]])

N,M = filter.shape
width,length = f.shape
print(width,length)
amount = int((width - N)/2)
print(amount)
amount2 = int((length - M)/2)
print(amount2)

if(N % 2 != 0 ):
    print("c")
    filter = np.pad(filter,[(amount,),(amount2,)], 'constant', constant_values=(0,0))
    c = [filter[0]]
    filter = np.append(c,filter,axis=0)
    filter = np.transpose(filter)
    c = [filter[0]]
    filter = np.append(c,filter,axis=0)
    filter = np.transpose(filter)
    
    
else:
    print("d")
    filter = np.pad(filter,[(amount,),(amount2,)], 'constant', constant_values=(0,0))
    filter = np.transpose(filter)
    c = [filter[0]]
    filter = np.append(c,filter,axis=0)
    filter = np.transpose(filter)

print(filter.shape)
print(filter)