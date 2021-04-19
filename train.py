import h5py
import os
import numpy as np
import random
import dataset
from torch.utils.data import DataLoader
import torch
from networks.UNet import Modified3DUNet
from networks.Discriminator import Discriminator
from loss import SegLoss, DisLoss, BinaryDiceLoss
import argparse
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

writer = SummaryWriter('/output/logs')


# Training settings
parser = argparse.ArgumentParser(description='GANbased-NeuralSegmentation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--g_lr', type=float, default=0.0001, help='initial generator learning rate for adam')
parser.add_argument('--d_lr', type=float, default=0.00002, help='initial discriminator learning rate for adam')
parser.add_argument('--alpha', type=float, default=0.6, help='the weight between GANLoss and entropy. Default=0.6')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
root_path = '/'
train_set_path = os.path.join(root_path, 'train_set')
prior_set_path = os.path.join(root_path, 'prior_set')
test_set_path = os.path.join(root_path, 'test_set')
train_set = dataset.TrainDataset(train_set_path, filenumber=11, sample_number=1000)
train_unpair_set = dataset.PriorDataset(prior_set_path, sample_number=1000)
test_set = dataset.TestDataset(test_set_path)
training_iterator = DataLoader(dataset=train_set, batch_size=opt.batch_size,  num_workers=opt.threads, shuffle=True)
train_unpair_iterator = DataLoader(dataset=train_unpair_set, batch_size=opt.batch_size,  num_workers=opt.threads, shuffle=True)
test_iterator = DataLoader(dataset=test_set, batch_size=opt.batch_size,  num_workers=opt.threads, shuffle=True)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
mydisnet = Discriminator(1, 1)
mysegnet = Modified3DUNet(1, 2)
mydisnet = torch.nn.DataParallel(mydisnet).cuda()
mysegnet = torch.nn.DataParallel(mysegnet).cuda()
  
criterionEntropy = torch.nn.CrossEntropyLoss().cuda()
criterionGANSeg = torch.nn.MSELoss().cuda()
criterionGANDis = torch.nn.MSELoss().cuda()
criterionDice = BinaryDiceLoss().cuda()
    
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)

optimizer_g = torch.optim.Adam(mysegnet.parameters(), lr=opt.g_lr, betas=(opt.beta1, 0.999))
optimizer_d = torch.optim.Adam(mydisnet.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

alpha = opt.alpha


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, ((batchX, batchY), (batchY1)) in enumerate(zip(training_iterator, train_unpair_iterator)):
        # forward
        # input data
        batchX = batchX.unsqueeze(1).to(device='cuda').float()
        batchY = batchY.unsqueeze(1).to(device='cuda').float()
        real_y = batchY.unsqueeze(1).to(device='cuda').float()
        real_y1 = batchY1.unsqueeze(1).to(device='cuda').float()
        # GAN label, soft label
        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
        soft_one = np.random.random_sample() * (1.2 - 0.8) + 0.8
        soft_zero = np.random.random_sample() * (0.3 - 0.0) + 0.0
        target_real = torch.autograd.Variable(Tensor(opt.batch_size).fill_(soft_one), requires_grad=False).cuda()
        target_fake = torch.autograd.Variable(Tensor(opt.batch_size).fill_(soft_zero), requires_grad=False).cuda()
        # G output
        groundtruth = batchY.view(-1).to(device='cuda', dtype=torch.int64)
        out, y_hat = mysegnet(batchX)
        fake_y = out[:,1].reshape([opt.batch_size, 1, 128, 128, 128]).clone().detach()
        # fake_y = torch.argmax(y_hat, dim=1, keepdim=True).float()
    
        ######################
        # (1) Update D network
        ######################
        
        optimizer_d.zero_grad()
        # add noise
        noisy1 = (0.1**0.5/(epoch+1))*torch.randn(real_y1.shape).cuda()
        noisy2 = (0.1**0.5/(epoch+1))*torch.randn(fake_y.shape).cuda()

        pred_real = mydisnet(real_y1 + noisy1)
        pred_fake = mydisnet(fake_y + noisy2)

        # train with fake and real, and transform the label
        loss_d = criterionGANDis(pred_fake, target_real) + criterionGANDis(pred_real, target_fake)
        # loss_d = -torch.mean(pred_real) + torch.mean(pred_fake)

        # Clip weights of discriminator
        # for p in mydisnet.parameters():
            # p.data.clamp_(-opt.clip_value, opt.clip_value)

        loss_d.backward()

        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################
        optimizer_g.zero_grad()

        # First, G(X) should fake the discriminator
        pred_fake = mydisnet(fake_y)
        loss_g_gan = criterionGANSeg(pred_fake, target_fake)
        # loss_g_gan = -torch.mean(pred_fake) * opt.alpha

        # Second, G(X) = Y
        loss_g_etp = criterionEntropy(out, groundtruth)
        # loss_g_dic = criterionDice(fake_y, batchY)
        
        # loss_g = loss_g_gan + loss_g_etp + loss_g_dic
        
        loss_g = alpha * loss_g_gan + (1 - alpha) * loss_g_etp
        
#         reverse_gt =  torch.cat((groundtruth, - groundtruth + 1), 0)
#         reverse_gt = reverse_gt.contiguous().view(-1, 2).float()
#         loss_g = 0.1 * criterionEntropy(out, groundtruth) + criterionEntropy(reverse_gt, out)

        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_iterator), loss_d.item(), loss_g.item()))
        
        if iteration % 10 == 0:
            niter = (epoch - 1) * len(training_iterator) + iteration
            writer.add_scalar('Train/Loss_g', loss_g.item(), niter)
#             writer.add_scalar('Train/Loss_d', loss_d.item(), niter)
#             writer.add_scalar('Train/Loss_g_gan', loss_g_gan.item(), niter)
#             writer.add_scalar('Train/Loss_g_seg', loss_g_etp.item(), niter)
            # writer.add_scalar('Train/Loss_g_dic', loss_g_dic.item(), niter)


    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
    avg_precission = 0.0
    avg_recall = 0.0
    for step, (batchX, batchY, _, _, _) in enumerate(test_iterator):
        batchX = batchX.unsqueeze(1).to(device='cuda').float()
        groundtruth = batchY.unsqueeze(1).to(device='cuda').float()

        # get the prediction using segmentation network
        out, y_hat = mysegnet(batchX)
        prediction = torch.argmax(y_hat, dim=1, keepdim=True).float()

        ons=torch.autograd.Variable(torch.ones(prediction.size(), device='cuda'))
        zes=torch.autograd.Variable(torch.zeros(prediction.size(), device='cuda'))
        train_correct11 = ((prediction==ons)&(groundtruth==ons)).sum()
        train_correct00 = ((prediction==zes)&(groundtruth==zes)).sum()
        train_correct10 = ((prediction==ons)&(groundtruth==zes)).sum()
        train_correct01 = ((prediction==zes)&(groundtruth==ons)).sum()
        rallone = (prediction==ons).sum()
        gallone = (groundtruth==ons).sum()
        precission = train_correct11.item()/rallone.item()
        recall = train_correct11.item()/gallone.item()
        
        niter = (epoch - 1) * len(test_iterator) + step
        writer.add_scalar('Test/Precission', precission, niter)
        writer.add_scalar('Test/Recall', recall, niter)

        avg_precission += precission
        avg_recall += recall
    print("===> Avg. Precission: {:.4f} | Avg. Recall: {:.4f}".format(avg_precission / len(test_iterator), avg_recall / len(test_iterator)))

    #checkpoint
    if epoch % 10 == 0:
        if not os.path.exists("/output/checkpoint"):
            os.mkdir("/output/checkpoint")
        if not os.path.exists(os.path.join("/output/checkpoint", opt.dataset)):
            os.mkdir(os.path.join("/output/checkpoint", opt.dataset))
        net_g_model_out_path = "/output/checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "/output/checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(mysegnet, net_g_model_out_path)
        torch.save(mydisnet, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
