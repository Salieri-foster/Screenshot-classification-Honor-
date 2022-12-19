import time
import torch

if __name__ == '__main__':
    print('torch版本：'+torch.__version__)
    print('cuda是否可用：'+str(torch.cuda.is_available()))
    print('cuda版本：'+str(torch.version.cuda))
    print('cuda数量:'+str(torch.cuda.device_count()))
    print('GPU名称：'+str(torch.cuda.get_device_name()))
    print('当前设备索引：'+str(torch.cuda.current_device()))

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    print(torch.rand(3, 3).cuda())

    for i in range(1,100000):
        start = time.time()
        a = torch.FloatTensor(i*100,1000,1000)
        a = a.cuda() #a = a
        a = torch.matmul(a,a)
        end = time.time() - start
        print(end)
