import torch
import torch.optim as optimizer
import os
import datetime
from functiondemo import *
from networkdemo import *
from torchvision import transforms
import numpy as np
import argparse


#
parse = argparse.ArgumentParser()

parse.add_argument('--input_dir', default='./image/', help='input image dir')
parse.add_argument('--train_dir', default='./trained/', help='slices in train process')
parse.add_argument('--model_dir', default='./model/', help='model parameter')
parse.add_argument('--check_dir', default='./checkpoint/', help='checkpoint model parameter')
parse.add_argument('--img', default='SampleB.bmp', help='given 2D img')

parse.add_argument('--scale_N', default=5, type=int, help='learning scale. default=5')
parse.add_argument('--lr_g', default=5e-4, type=float, help='generator learning rate')
parse.add_argument('--lr_d', default=3e-4, type=float, help='discriminator learning rate')
parse.add_argument('--beta', type=float, default=0.9, help='beta1 for adam. default=0.9')
parse.add_argument('--gp_lambda', type=float, default=8, help='gradient penalty weight')
parse.add_argument('--gamma', type=float, help='scheduler gamma', default=0.9)
parse.add_argument('--batch_size', default=1, type=int, help='generator batch_size')
parse.add_argument('--batch_slice', default=16, type=int, help='slice batch_size')
parse.add_argument('--direction_w', default=32, type=int, help='size of 3D image on training direction,multiples of 8')
parse.add_argument('--cuda', default='cuda:0', help='cuda number')
parse.add_argument('--pre', default=False, help='yes or no fix pre-training parameters')
parse.add_argument('--is_check', default=False, help='yes or no use previous checkpoint train')


# tscale and cha are not fixed, it needs to be set according to the size of the examples
parse.add_argument('--tscale', default=304, type=int, help='max crop size, multiples of 8')
parse.add_argument('--cha', default=16, type=int, help='interval of multi cropped size, multiples of 8')

parse.add_argument('--vggstep', default=50000, type=int, help='perceptual loss')
parse.add_argument('--vggk', default=0.01, type=float, help='the coefficient of perceptual loss')

parse.add_argument('--iter_train', default=90000, type=int, help='total train iter, 50000-120000')
parse.add_argument('--iter_slice', default=100, type=int, help='every num save train slice')
parse.add_argument('--iter_check', default=10000, type=int, help='every num save checkpoint')
parse.add_argument('--iter_model', default=1000, type=int, help='every num save model')
parse.add_argument('--step_channel', default=8, type=int, help='step_channel')


opt = parse.parse_args()

# device
device = torch.device(opt.cuda)
batch_size = opt.batch_size

# read image
input_image = opt.img
opt.input_dir = opt.input_dir + input_image
image = load_image(opt.input_dir)


# image info
opt.channel = len(image.split())
opt.real_img_h = image.size[0]
opt.real_img_w = image.size[1]
print(opt.channel, opt.real_img_h, opt.real_img_w)


# create trained dir
time_now = datetime.datetime.now()
opt.train_dir = opt.train_dir + time_now.strftime("%Y-%m-%d-%H-%M-%S-") + input_image[0:input_image.rfind('.')]
os.mkdir(opt.train_dir)


# create model dir
opt.model_dir = opt.model_dir + time_now.strftime("%Y-%m-%d-%H-%M-%S-") + input_image[0:input_image.rfind('.')]
os.mkdir(opt.model_dir)
# create check dir
opt.check_dir = opt.check_dir + time_now.strftime("%Y-%m-%d-%H-%M-%S-") + input_image[0:input_image.rfind('.')]
os.mkdir(opt.check_dir)


# load vgg19_model
vgg = VGG(pool='avg', pad=1)
vgg.load_state_dict(torch.load('./code/vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
vgg.to(device)


# Perceptual Loss
loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
loss_fns = [GramMSELoss()] * len(loss_layers)
loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]
wd = [1, 1, 1, 1, 1]


# setup generator network
net_G = test_Generator3(opt.channel, opt.step_channel)
net_G.apply(weights_init)
net_G.to(device)
optimizer_G = optimizer.Adam(net_G.parameters(), lr=opt.lr_g, betas=(opt.beta, 0.999))

# setup discriminators network && load vgg19_model
net_Ds = []
optimizer_Ds = []
for i in range(opt.scale_N):
    vgg_dict = torch.load('./code/vgg_conv.pth')
    net_D = VD(in_channel=3, pad=1)
    net_D.apply(weights_init)
    net_D_dict = net_D.state_dict()
    vgg_dict = {k: v for k, v in vgg_dict.items() if k in net_D_dict}
    net_D_dict.update(vgg_dict)
    net_D.load_state_dict(net_D_dict)
    for name, param in net_D.named_parameters():
        if name.find('conv') != -1:
            param.requires_grad = opt.pre
    optimizer_D = optimizer.Adam(net_D.parameters(), lr=opt.lr_d, betas=(opt.beta, 0.999))
    net_D.to(device)
    net_Ds.append(net_D)
    optimizer_Ds.append(optimizer_D)

# save train config
save_config(opt, opt.train_dir)


# train
if __name__ == '__main__':
    Gloss_history = np.zeros((opt.iter_train))
    Dloss_history = np.zeros((opt.scale_N, opt.iter_train))
    Vloss_history = np.zeros((opt.iter_train))

    tt, t = opt.tscale, opt.cha
    h_rand = [tt-t*4, tt-t*3, tt-t*2, tt-t, tt]
    time_now = datetime.datetime.now() 
    # print('start',time_now)
    strat = 0
    if opt.is_check == True:
        filepath = 'checkpoint/XXXX_checkpoint.pth'
        strat = load_checkpoint(filepath, net_G, net_Ds, optimizer_G, optimizer_Ds, device) + 1
        
    for iter_num in range(strat, opt.iter_train):
        if (iter_num+1) % 5000 == 0:
            opt.gp_lambda = opt.gp_lambda * 0.9

        scale_choice = np.random.randint(0,opt.scale_N)
        h = h_rand[scale_choice]
        w = opt.direction_w
        fake_slices = []

        for d in range(3):
            if d == 0:
                input_rand = [torch.randn([opt.batch_size, 3, int(w/8 + 10), int(h/8 + 10), int(h/8 + 10)], device=device),
                              torch.randn([opt.batch_size, 3, int(w/4 + 14), int(h/4 + 14), int(h/4 + 14)], device=device),
                            torch.randn([opt.batch_size, 3, int(w/2 + 14), int(h/2 + 14), int(h/2 + 14)], device=device),
                            torch.randn([opt.batch_size, 3, int(w+14), int(h+14), int(h+14)], device=device)]
                fake = net_G(input_rand)
                fake = F.interpolate(fake, size=[w, h_rand[-3], h_rand[-3]], mode='trilinear')
                fake_slice = cutone_d0(fake, opt.batch_slice).to(device)
                fake_slices.append(fake_slice)

            if d == 1:
                input_rand = [torch.randn([opt.batch_size, 3, int(h/8 + 10), int(w/8 + 10), int(h/8 + 10)], device=device),
                              torch.randn([opt.batch_size, 3, int(h/4 + 14), int(w/4 + 14), int(h/4 + 14)], device=device),
                            torch.randn([opt.batch_size, 3, int(h/2 + 14), int(w/2 + 14), int(h/2 + 14)], device=device),
                            torch.randn([opt.batch_size, 3, int(h+14), int(w+14), int(h+14)], device=device)]
                fake = net_G(input_rand)
                fake = F.interpolate(fake, size=[h_rand[-3], w, h_rand[-3]], mode='trilinear')
                fake_slice = cutone_d1(fake, opt.batch_slice).to(device)
                fake_slices.append(fake_slice)
                
            if d == 2:
                input_rand = [torch.randn([opt.batch_size, 3, int(h/8 + 10), int(h/8 + 10), int(w/8 + 10)], device=device),
                              torch.randn([opt.batch_size, 3, int(h/4 + 14), int(h/4 + 14), int(w/4 + 14)], device=device),
                            torch.randn([opt.batch_size, 3, int(h/2 + 14), int(h/2 + 14), int(w/2 + 14)], device=device),
                            torch.randn([opt.batch_size, 3, int(h+14), int(h+14), int(w+14)], device=device)]
                fake = net_G(input_rand)
                fake = F.interpolate(fake, size=[h_rand[-3], h_rand[-3], w], mode='trilinear')
                fake_slice = cutone_d2(fake, opt.batch_slice).to(device)
                fake_slices.append(fake_slice)

        fake_slices = torch.cat(fake_slices, dim=0)

        optimizer_G.zero_grad()
        if (iter_num + 1) % 10 == 0 and (iter_num + 1) <= opt.vggstep:
            # real patches
            texture = real_rand_cropopt(image, h, 3*batch_size * opt.batch_slice, h_rand[-3], device)
            texture = (texture)*255.
            # real patches gram
            targets = [GramMatrix()(f).detach() for f in vgg(texture, loss_layers)]
            # fake slices gram
            fake_slices_gram = vgg(de_norm(fake_slices)*255, loss_layers)
            # gram loss
            vgg_loss = opt.vggk*sum([wd[a] * loss_fns[a](f, targets[a]) for a, f in enumerate(fake_slices_gram)])
            Vloss_history[iter_num] = Vloss_history[iter_num] + vgg_loss.item()
            vgg_loss.backward(retain_graph = True)
            
            
        # Generator train
        optimizer_Ds[scale_choice].zero_grad()
        optimizer_G.zero_grad()
        out_fake = net_Ds[scale_choice](fake_slices)
        errG = -out_fake.mean()
        errG.backward()
        errorG = errG.data
        Gloss_history[iter_num] = Gloss_history[iter_num] + errorG.item()
        optimizer_G.step()

        del fake_slices, input_rand

        # multi discriminators train
        for s in range(opt.scale_N):
            h = h_rand[s]
            w = opt.direction_w
            fake_slices = []
            for d in range(3):
                if d == 0:
                    input_rand = [torch.randn([opt.batch_size, 3, int(w/8 + 10), int(h/8 + 10), int(h/8 + 10)], device=device),
                                torch.randn([opt.batch_size, 3, int(w/4 + 14), int(h/4 + 14), int(h/4 + 14)], device=device),
                                torch.randn([opt.batch_size, 3, int(w/2 + 14), int(h/2 + 14), int(h/2 + 14)], device=device),
                                torch.randn([opt.batch_size, 3, int(w+14), int(h+14), int(h+14)], device=device)]
                    fake = net_G(input_rand)
                    fake = F.interpolate(fake, size=[w, h_rand[-3], h_rand[-3]], mode='trilinear')
                    fake_slice = cutone_d0(fake, opt.batch_slice).to(device)
                    fake_slices.append(fake_slice)

                if d == 1:
                    input_rand = [torch.randn([opt.batch_size, 3, int(h/8 + 10), int(w/8 + 10), int(h/8 + 10)], device=device),
                                torch.randn([opt.batch_size, 3, int(h/4 + 14), int(w/4 + 14), int(h/4 + 14)], device=device),
                                torch.randn([opt.batch_size, 3, int(h/2 + 14), int(w/2 + 14), int(h/2 + 14)], device=device),
                                torch.randn([opt.batch_size, 3, int(h+14), int(w+14), int(h+14)], device=device)]
                    fake = net_G(input_rand)
                    fake = F.interpolate(fake, size=[h_rand[-3], w, h_rand[-3]], mode='trilinear')
                    fake_slice = cutone_d1(fake, opt.batch_slice).to(device)
                    fake_slices.append(fake_slice)
                    
                if d == 2:
                    input_rand = [torch.randn([opt.batch_size, 3, int(h/8 + 10), int(h/8 + 10), int(w/8 + 10)], device=device),
                                torch.randn([opt.batch_size, 3, int(h/4 + 14), int(h/4 + 14), int(w/4 + 14)], device=device),
                                torch.randn([opt.batch_size, 3, int(h/2 + 14), int(h/2 + 14), int(w/2 + 14)], device=device),
                                torch.randn([opt.batch_size, 3, int(h+14), int(h+14), int(w+14)], device=device)]
                    fake = net_G(input_rand)
                    fake = F.interpolate(fake, size=[h_rand[-3], h_rand[-3], w], mode='trilinear')
                    fake_slice = cutone_d2(fake, opt.batch_slice).to(device)
                    fake_slices.append(fake_slice)
            
            fake_slices = torch.cat(fake_slices, dim=0)

            texture = real_rand_cropopt(image, h,  3 * batch_size * opt.batch_slice, h_rand[-3], device)
            real = norm(texture)
            real = real.to(device)

            # optimize all scale D
            optimizer_Ds[s].zero_grad()
            real_out = net_Ds[s](real)
            D_real = -real_out.mean()
            D_real.backward()

            fake_out = net_Ds[s](fake_slices.detach())
            D_fake = fake_out.mean()
            D_fake.backward()

            gp = gradient_penalty(net_Ds[s], real, fake_slices.detach(), opt.gp_lambda, device)
            gp.backward()
            Dloss_history[s][iter_num] = Dloss_history[s][iter_num] + D_real.item()+D_fake.item()+gp.item()
            optimizer_Ds[s].step()
            del input_rand, fake_slices

        # loss show
        if (iter_num + 1) % 10 == 0:
            print('iter:', iter_num,  ', D0:', format(Dloss_history[0][iter_num], '.3f'),', D1:', format(Dloss_history[1][iter_num], '.2f'),', D2:', format(Dloss_history[2][iter_num], '.2f'),', D3:', format(Dloss_history[3][iter_num], '.2f'),', D4:', format(Dloss_history[4][iter_num], '.2f'), ', G:', format(Gloss_history[iter_num], '.2f'), ', VGGloss:', format(Vloss_history[iter_num], '.4f'))

        # train process show results
        if (iter_num + 1) % opt.iter_slice == 0:
            # print(datetime.datetime.now())
            net_G.eval()
            with torch.no_grad():
                w = 8
                h = 400
                for dd in range(3):
                    if dd == 0:
                        a = [torch.randn([opt.batch_size, 3, int(w/8 + 10), int(h/8 + 10), int(h/8 + 10)], device=device),
                                    torch.randn([opt.batch_size, 3, int(w/4 + 14), int(h/4 + 14), int(h/4 + 14)], device=device),
                                    torch.randn([opt.batch_size, 3, int(w/2 + 14), int(h/2 + 14), int(h/2 + 14)], device=device),
                                    torch.randn([opt.batch_size, 3, int(w+14), int(h+14), int(h+14)], device=device)]
                        
                    if dd == 1:
                        a = [torch.randn([opt.batch_size, 3, int(h/8 + 10), int(w/8 + 10), int(h/8 + 10)], device=device),
                                    torch.randn([opt.batch_size, 3, int(h/4 + 14), int(w/4 + 14), int(h/4 + 14)], device=device),
                                    torch.randn([opt.batch_size, 3, int(h/2 + 14), int(w/2 + 14), int(h/2 + 14)], device=device),
                                    torch.randn([opt.batch_size, 3, int(h+14), int(w+14), int(h+14)], device=device)]
                        
                    if dd == 2:
                        a = [torch.randn([opt.batch_size, 3, int(h/8 + 10), int(h/8 + 10), int(w/8 + 10)], device=device),
                                    torch.randn([opt.batch_size, 3, int(h/4 + 14), int(h/4 + 14), int(w/4 + 14)], device=device),
                                    torch.randn([opt.batch_size, 3, int(h/2 + 14), int(h/2 + 14), int(w/2 + 14)], device=device),
                                    torch.randn([opt.batch_size, 3, int(h+14), int(h+14), int(w+14)], device=device)]

                    im = net_G(a)
                    im = de_norm(im)
                    test_slice = fake_rand_crop(im, 1, 1, dd, w, opt.cuda).to(device)
                    test_slice = test_slice.squeeze(0).cpu()
                    test_slice = toPIL(test_slice).convert('L').convert('RGB')
                    test_slice.save(
                        opt.train_dir + '/' + str(iter_num).zfill(
                            5) + '_' + 'd' + str(dd) + '.bmp')
                del a, im, test_slice
            net_G.train()


        if (iter_num + 1) % opt.iter_model == 0 and (iter_num + 1) >= 20000:
            checkpoint_G = {'model': net_G.state_dict(), 'optimizer': optimizer_G.state_dict(), 'epoch': iter_num}
            torch.save(checkpoint_G, opt.model_dir + '/' + str(iter_num) + '_' + 'g_model.pth')
        if (iter_num + 1) % opt.iter_check == 0:
            save_checkpoint(net_G, net_Ds, optimizer_G, optimizer_Ds, iter_num, opt.check_dir + '/' + str(iter_num) + '_' + 'checkpoint.pth')




