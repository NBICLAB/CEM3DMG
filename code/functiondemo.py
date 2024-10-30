from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

totensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage() 

def load_image(image_dir):
    image = Image.open(image_dir).convert('RGB')
    return image


def normalization(image):
    t = transforms.Compose([
        transforms.ToTensor(),
    ])
    return t(image)


def norm(image):  # (0,1)->(-1,1)
    out = (image - 0.5) * 2
    return out.clamp(-1, 1)


def de_norm(image):  # (-1,1)->(0,1)
    out = (image + 1) / 2
    return out.clamp(0, 1)



def gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gp


def real_rand_cropopt(input, size, batch_size, re_size, device):
    rand_com = transforms.RandomCrop(size=size, padding=0, padding_mode='edge')
    rand_choice = transforms.RandomChoice([
        transforms.RandomRotation((90, 90)),
        transforms.RandomRotation((180, 180)),
        transforms.RandomRotation((270, 270)),
        transforms.RandomRotation((360, 360)),
        transforms.RandomHorizontalFlip(p=0.99),
        transforms.RandomVerticalFlip(p=0.99)
    ])
    resize = transforms.Resize((re_size, re_size), Image.BILINEAR)
    totensor = transforms.ToTensor()

    transformed_images = []

    for _ in range(batch_size):
        transformed_image = rand_com(input)
        transformed_image = rand_choice(transformed_image)
        transformed_image = resize(transformed_image)
        transformed_image = totensor(transformed_image)
        transformed_images.append(transformed_image)


    batch_tensor = torch.stack(transformed_images).to(device)

    return batch_tensor





def save_config(inputs, path):
    var = vars(inputs).items()
    f = open(path + '/config.txt', mode='w')
    for k, v in var:
        f.write(str(k) + ' : ')
        f.write(str(v) + '\n')
    f.close()


def cutone_d0(input, bs_slice):
    indices = torch.randperm(input.size(2))[:bs_slice]
    selected_data = input[:,:,indices,::]
    return selected_data.squeeze(0).permute(1,0,2,3)

def cutone_d1(input, bs_slice):
    indices = torch.randperm(input.size(3))[:bs_slice]
    selected_data = input[:,:,:,indices,:]
    return selected_data.squeeze(0).permute(2,0,1,3)

def cutone_d2(input, bs_slice):
    indices = torch.randperm(input.size(4))[:bs_slice]
    selected_data = input[:,:,:,:,indices]
    return selected_data.squeeze(0).permute(3,0,1,2)

def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, epoch, filepath):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator1_state_dict': discriminator[0].state_dict(),
        'discriminator2_state_dict': discriminator[1].state_dict(),
        'discriminator3_state_dict': discriminator[2].state_dict(),
        'discriminator4_state_dict': discriminator[3].state_dict(),
        'discriminator5_state_dict': discriminator[4].state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d1_state_dict': optimizer_d[0].state_dict(),
        'optimizer_d2_state_dict': optimizer_d[1].state_dict(),
        'optimizer_d3_state_dict': optimizer_d[2].state_dict(),
        'optimizer_d4_state_dict': optimizer_d[3].state_dict(),
        'optimizer_d5_state_dict': optimizer_d[4].state_dict(),
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, generator, discriminator, optimizer_g, optimizer_d, device):

    checkpoint = torch.load(filepath, map_location=device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator[0].load_state_dict(checkpoint['discriminator1_state_dict'])
    discriminator[1].load_state_dict(checkpoint['discriminator2_state_dict'])
    discriminator[2].load_state_dict(checkpoint['discriminator3_state_dict'])
    discriminator[3].load_state_dict(checkpoint['discriminator4_state_dict'])
    discriminator[4].load_state_dict(checkpoint['discriminator5_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d[0].load_state_dict(checkpoint['optimizer_d1_state_dict'])
    optimizer_d[1].load_state_dict(checkpoint['optimizer_d2_state_dict'])
    optimizer_d[2].load_state_dict(checkpoint['optimizer_d3_state_dict'])
    optimizer_d[3].load_state_dict(checkpoint['optimizer_d4_state_dict'])
    optimizer_d[4].load_state_dict(checkpoint['optimizer_d5_state_dict'])
    
    return checkpoint['epoch']




def fake_rand_crop(inputs, batch_size, batch_slice, d, test_witch, device):

    temp = []
    t = torch.zeros(1).to(device)
    rand_d = d
    for i in range(batch_size):
        input = inputs[i:i + 1, ::]
        for _ in range(batch_slice):
            rand_slice = random.randint(0, test_witch - 1)
            if rand_d == 0:
                t = input[:, :, rand_slice, :, :]
            if rand_d == 1:
                t = input[:, :, :, rand_slice, :]
            if rand_d == 2:
                t = input[:, :, :, :, rand_slice]

            temp.append(t)

    return torch.cat(temp, dim=0)


# calculate GramLoss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w * c)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


# VGG19
class VGG(nn.Module):
    def __init__(self, pool='max', pad=1):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=pad)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=pad)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=pad)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=pad)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
