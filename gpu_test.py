import torch
print(torch.cuda.is_available()) # True

X_train = torch.FloatTensor([0., 1., 2.])
X_train = X_train.cuda()
print(X_train.is_cuda) # True
print(torch.cuda.current_device())  # 0
print(torch.cuda.device_count()) # 1
print(torch.cuda.get_device_name(0)) #NVIDIA GeForce RTX 3060