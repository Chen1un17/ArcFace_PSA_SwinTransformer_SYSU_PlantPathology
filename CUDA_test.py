import torch
torch.cuda.is_available()
# 返回True 接着用下列代码进一步测试
torch.zeros(1).cuda()