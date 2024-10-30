import torch
import os
from functiondemo import *
from networkdemo import *
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def de_norm(image):  # (-1,1)->(0,1)
    out = (image + 1) / 2
    return out.clamp(0, 1)


def cubeslice(x, dim, path, h, w):
    hwd = x.shape[2:5]
    for i in range(hwd[dim]):
        slice = x[0, :, :, :, i]
        slice = slice.transpose(1, 2, 0)
        img = Image.fromarray(np.uint8(slice))
        img = img.convert('L')
        img = img.convert('RGB')
        img.save(path + '/' + str(h*w+i).zfill(4) + '.bmp')


if __name__ == '__main__':
    h = 400  # generated 3D image size, multiples of 8
    w = 8 # multiples of 8
    
    # create output dir
    path1 = './output/Test2'

    if os.path.exists(path1):
        print('Folder already exists')
    else:
        os.mkdir(path1)
        print('Create a folder')


    path_g = './model/Example.pth'
    net_G = test_Generator3(3, 8)
    net_G = net_G.to(device)
    # checkpoint_G = torch.load(path_g, map_location='cuda:0')
    checkpoint_G = torch.load(path_g, map_location=device)
    net_G.load_state_dict(checkpoint_G['model'])
    net_G.eval()

    solid = []
    input_rand = [torch.randn([1, 3, int(h/8 + 10), int(h/8 + 10), int(w/8 + 10)], device=device),
                torch.randn([1, 3, int(h/4 + 14), int(h/4 + 14), int(w/4 + 14)], device=device),
                torch.randn([1, 3, int(h/2 + 14), int(h/2 + 14), int(w/2 + 14)], device=device),
                torch.randn([1, 3, int(h+14), int(h+14), int(w+14)], device=device)]

    
    for i in range(int(h/w)):
        test1 = [torch.randn([1, 3, int(h/8 + 10), int(h/8 + 10), int(w/8 + 10)], device=device),
                    torch.randn([1, 3, int(h/4 + 14), int(h/4 + 14), int(w/4 + 14)], device=device),
                    torch.randn([1, 3, int(h/2 + 14), int(h/2 + 14), int(w/2 + 14)], device=device),
                    torch.randn([1, 3, int(h+14), int(h+14), int(w+14)], device=device)]


        test1[0][:,:,:,:,:10] = input_rand[0][:,:,:,:,-10:]
        test1[1][:,:,:,:,:14] = input_rand[1][:,:,:,:,-14:]
        test1[2][:,:,:,:,:14] = input_rand[2][:,:,:,:,-14:]
        test1[3][:,:,:,:,:14] = input_rand[3][:,:,:,:,-14:]


        input_rand = test1

        with torch.no_grad():
            im = net_G(test1)
            im = de_norm(im).cpu().numpy()*255
            cubeslice(im, -1, path1, i, w)

