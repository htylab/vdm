import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

import cv2
import glob
import warnings
warnings.filterwarnings("ignore")


fold_idx = int(sys.argv[1])
fold_num = int(sys.argv[2])
batch_size = int(sys.argv[3])
if sys.argv[4].find('-')==-1:
    start_epochs = 0
    num_epochs = int(sys.argv[4])
else:
    start_epochs = int(sys.argv[4].split('-')[0])
    num_epochs = int(sys.argv[4].split('-')[1])
workers = int(sys.argv[5])

gpu = '0'
# batch_size = 1
eps = 1e-5
seed = 168
num_trainG = 3
# num_epochs = 100
save_frq = 5
criterionGAN = nn.MSELoss()
criterionL1 = nn.L1Loss()
real_label=torch.tensor(1)
fake_label=torch.tensor(0)
lambda_L1 = 100.0
lr = 1e-4
weight_decay = 1e-5
amsgrad = True
# workers = 0


import models
Network_G = getattr(models, 'UNet3d')
Network_D = getattr(models, 'UNet3d')

path = 'dataset/MPI-LEMON_MRI'

save_dir = f'result'


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

seed_everything(seed)

seed_dir = os.path.join(save_dir, f'EP{num_epochs}/fold_{fold_idx}')

os.makedirs(seed_dir, exist_ok=True)

os.environ['CUDA_VISIBLE_DEVICES'] = gpu
assert torch.cuda.is_available(), "Currently, we only support CUDA version"
device = (f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')



class Getdata():
    def __init__(self, names, mode = 'test'):
        
        self.names = names
        self.mode = mode
        
        self.epi_PA_ffs = []
        self.field_PA_ffs = []
        for name in self.names:
            ss = sorted(glob.glob(os.path.join(path, 'raw', 'data', name, 'ses-01', 'fmap', '*SEfmapDWI*PA_epi.nii.gz')))
            self.epi_PA_ffs.append(ss[0])
            ss = sorted(glob.glob(os.path.join(path, 'fout', name, 'ses-01', 'fmap', 'filed_PA_out.nii.gz')))
            self.field_PA_ffs.append(ss[0])
            
        self.epi_AP_ffs = []
        self.field_AP_ffs = []
        for name in self.names:
            ss = sorted(glob.glob(os.path.join(path, 'raw', 'data', name, 'ses-01', 'fmap', '*SEfmapDWI*AP_epi.nii.gz')))
            self.epi_AP_ffs.append(ss[0])
            ss = sorted(glob.glob(os.path.join(path, 'fout', name, 'ses-01', 'fmap', 'filed_AP_out.nii.gz')))
            self.field_AP_ffs.append(ss[0])
        
        self.t2_ffs = []
        for name in self.names:
            ss = sorted(glob.glob(os.path.join(path, 'topup_result', name, 'ses-01', 'reg', '*t22topup_reg.nii.gz')))
            self.t2_ffs.append(ss[0])
                
        self.t1_ffs = []
        for name in self.names:
            ss = sorted(glob.glob(os.path.join(path, 'topup_result', name, 'ses-01', 'reg', '*t12topup_reg.nii.gz')))
            self.t1_ffs.append(ss[0])
            
        self.bet_ffs = []
        for name in self.names:
            ss = sorted(glob.glob(os.path.join(path, 'topup_result', name, 'ses-01', 'bet', '*mask.nii.gz')))
            self.bet_ffs.append(ss[0])
        
    def __len__(self):
        return len(self.t1_ffs)*2
    
    def __getitem__(self, idx):
        
        if idx % 2 == 0:
            idx = idx//2
            
            temp = nib.load(self.epi_AP_ffs[idx])
            epi_image = temp.get_fdata()[...,0]
            field_image = nib.load(self.field_AP_ffs[idx]).get_fdata()

            epi_image = (np.array(epi_image, dtype='float32') / np.max(epi_image))
            field_image = (np.array(field_image, dtype='float32'))# / np.max(epi_down_image)

            epi_image = torch.from_numpy(epi_image[np.newaxis,...])
            field_image = torch.from_numpy(field_image[np.newaxis,...])
            
            return {'epi': epi_image, 
                    'field': field_image, 
                    'path': self.epi_AP_ffs[idx], 
                    'name': self.names[idx], 
                    'affine': temp.affine}
            
        else:
            idx = idx//2
            
            temp = nib.load(self.epi_PA_ffs[idx])
            epi_image = temp.get_fdata()[...,0]
            field_image = nib.load(self.field_PA_ffs[idx]).get_fdata()

            epi_image = (np.array(epi_image, dtype='float32') / np.max(epi_image))
            field_image = (np.array(field_image, dtype='float32'))# / np.max(epi_down_image)

            epi_image = torch.from_numpy(epi_image[np.newaxis,...])
            field_image = torch.from_numpy(field_image[np.newaxis,...])
            
            return {'epi': epi_image, 
                    'field': field_image, 
                    'path': self.epi_PA_ffs[idx], 
                    'name': self.names[idx], 
                    'affine': temp.affine}


dName = dict()

remove = ['sub-032308', 'sub-032313', 'sub-032344']

ffs = sorted(glob.glob(os.path.join(path, 'topup_result', '*')))
temp = []
for i in range(len(ffs)):
    ff = os.path.split(ffs[i])[1]
    if ff not in remove:
        temp.append(ff)
temp = sorted(list(set(temp)))



print('train+test', len(temp))
random.shuffle(temp)

sample = len(temp) // fold_num
fold = []
for jj in range(fold_num-1):
    fold.append(temp[:sample])
    del temp[:sample]
fold.append(temp)

dName[f'epi_fold'] = fold.copy()


dataset = dict()
dataset['test'] = dName[f'epi_fold'][fold_idx].copy()

temp = []
for pid in dName[f'epi_fold']:
    if not (pid == dataset['test']):
        temp += pid
dataset[f'train'] = temp

# dataset[f'train'] = dataset[f'train'][:2]
# dataset[f'test'] = dataset[f'test'][:2]
    
#check fold splitting
print('Loading data...')

dGetdata = dict()
for str1 in ['test', 'train']:
    dGetdata[str1] = Getdata(dataset[str1], str1)
    print(str1, len(dataset[str1]), len(dGetdata[str1]))


# %%

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val_all = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val_all.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = np.std(np.array(self.val_all))


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
#         param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
        param_group['lr'] = INIT_LR

def plot(epoch, tGloss_plt, tDloss_plt, vGloss_plt, vDloss_plt):
    plt.figure()
    plt.plot(tGloss_plt,'-', label='Train')
    plt.plot(vGloss_plt,'-', label='Valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('G Train vs Valid Loss')
 
    plt.savefig(os.path.join(seed_dir, f'picture/epoch{epoch} G Train vs Valid Loss.png'))  

    plt.figure()
    plt.plot(tDloss_plt,'-', label='Train')
    plt.plot(vDloss_plt,'-', label='Valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('D Train vs Valid Loss')
 
    plt.savefig(os.path.join(seed_dir, f'picture/epoch{epoch} D Train vs Valid Loss.png'))  

    plt.close('all')


# %%


os.makedirs(os.path.join(seed_dir, 'ckpts'), exist_ok=True)
os.makedirs(os.path.join(seed_dir, 'picture'), exist_ok=True)

model_G = Network_G(in_channels=1, n_classes=1, n_channels=24, z_pooling=True).to(device)
model_D = Network_D(in_channels=2, n_classes=1, n_channels=24, z_pooling=True).to(device)
if start_epochs!=0:
    model_G.load_state_dict(torch.load(os.path.join(save_dir, f'EP{start_epochs}/fold_{fold_idx}', f'ckpts/modelG_epoch_{start_epochs}.pt')))
    model_D.load_state_dict(torch.load(os.path.join(save_dir, f'EP{start_epochs}/fold_{fold_idx}', f'ckpts/modelD_epoch_{start_epochs}.pt')))
model_VGG = torchvision.models.vgg19(pretrained=True).to(device)
optimizer_G = torch.optim.Adam(model_G.parameters(), lr = lr, weight_decay= weight_decay, amsgrad = amsgrad)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr = lr, weight_decay= weight_decay, amsgrad = amsgrad)

print(f'Cross Set Number : {fold_idx}')
print("-------------- New training session ----------------")

train_set = dGetdata['train']
valid_set = dGetdata['test']

train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle = True,
        num_workers=workers,
        pin_memory=True)

valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=1,
        shuffle = False,
        num_workers=workers)

tGloss_plt = []
tDloss_plt = []
tGlosses = AverageMeter()
tDlosses = AverageMeter()

vGloss_plt = []
vDloss_plt = []
vGlosses = AverageMeter()
vDlosses = AverageMeter()

for j in tqdm(range(start_epochs,num_epochs,1), total=num_epochs, initial=start_epochs+1, 
                      desc=f'fold{fold_idx}_Epoch', bar_format='{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]\n'):
#     print(f"Epoch:{j+1}/{num_epochs}")
    print('Train')
    model_G.train()
    model_D.train()
    for data in tqdm(train_loader):
        adjust_learning_rate(optimizer_G, j+1, num_epochs, lr)
        adjust_learning_rate(optimizer_D, j+1, num_epochs, lr)

        real_A = data['epi'].to(device)
        real_B = data['field'].to(device)

        # First, G(A) should fake the discriminator
        for t in range(num_trainG):
            fake_B = model_G(real_A)

            optimizer_G.zero_grad()
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = model_D(fake_AB)
            target_tensor = real_label
            target_tensor = target_tensor.expand_as(pred_fake).to(device)
            loss_G_GAN = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
            # Second, G(A) = B
            loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
            # Third, Feature(G(A)) = Feature(B)
            loss_G_feature = criterionGAN(model_VGG(torch.cat([fake_B, fake_B, fake_B], dim=1).squeeze().permute(3, 0, 1, 2)), 
                                          model_VGG(torch.cat([real_B, real_B, real_B], dim=1).squeeze().permute(3, 0, 1, 2)))

            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_feature
            loss_G.backward()
            optimizer_G.step()


        optimizer_D.zero_grad()
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((real_A, fake_B), 1) 
        pred_fake = model_D(fake_AB.detach())
        target_tensor = fake_label
        target_tensor = target_tensor.expand_as(pred_fake).to(device)
        loss_D_fake = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = model_D(real_AB)
        target_tensor = real_label
        target_tensor = target_tensor.expand_as(pred_real).to(device)
        loss_D_real = criterionGAN(pred_real.to(torch.float32), target_tensor.to(torch.float32))

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optimizer_D.step()

        tGlosses.update(loss_G.item(), batch_size)
        tDlosses.update(loss_D.item(), batch_size)

    print()
    print('-----------------------Train-----------------------')
    print('G Loss {:.7f}'.format(tGlosses.avg))
    print('D Loss {:.7f}'.format(tDlosses.avg))
    print('---------------------------------------------------')

    tGloss_plt.append(tGlosses.avg)
    tDloss_plt.append(tDlosses.avg)
    tGlosses.reset()
    tDlosses.reset()


    # if ((j+1) % 5 == 0 ) & (j != 1):
    print('Vaild')
    model_G.eval()
    model_D.eval()
    for i, data in enumerate(tqdm(valid_loader)):
        real_A = data['epi'].to(device)
        real_B = data['field'].to(device)

        fake_B = model_G(real_A)

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = model_D(fake_AB)
        target_tensor = real_label
        target_tensor = target_tensor.expand_as(pred_fake).to(device)
        loss_G_GAN = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
        # Second, G(A) = B
        loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
        # Third, Feature(G(A)) = Feature(B)
        loss_G_feature = criterionGAN(model_VGG(torch.cat([fake_B, fake_B, fake_B], dim=1).squeeze().permute(3, 0, 1, 2)),
                                      model_VGG(torch.cat([real_B, real_B, real_B], dim=1).squeeze().permute(3, 0, 1, 2)))

        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1 + loss_G_feature

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((real_A, fake_B), 1) 
        pred_fake = model_D(fake_AB.detach())
        target_tensor = fake_label
        target_tensor = target_tensor.expand_as(pred_fake).to(device)
        loss_D_fake = criterionGAN(pred_fake.to(torch.float32), target_tensor.to(torch.float32))
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = model_D(real_AB)
        target_tensor = real_label
        target_tensor = target_tensor.expand_as(pred_real).to(device)
        loss_D_real = criterionGAN(pred_real.to(torch.float32), target_tensor.to(torch.float32))

        loss_D = (loss_D_fake + loss_D_real) * 0.5


        if (j+1) % save_frq == 0 and i<5:
            name = os.path.split(data['path'][0])[1].split(".nii.gz")[0]
            os.makedirs(os.path.join(seed_dir, f'picture_{j+1}'), exist_ok=True)
            fake_B_np = np.squeeze(fake_B.to('cpu').detach().numpy())
            real_B_np = np.squeeze(real_B.to('cpu').detach().numpy())
            nib.save(nib.Nifti1Image(fake_B_np, data['affine'][0].detach().numpy()), os.path.join(seed_dir, f"picture_{j+1}", f"{name}_fake.nii.gz"))
            nib.save(nib.Nifti1Image(real_B_np, data['affine'][0].detach().numpy()), os.path.join(seed_dir, f"picture_{j+1}", f"{name}_real.nii.gz"))


        vGlosses.update(loss_G.item(), batch_size)
        vDlosses.update(loss_D.item(), batch_size)

    print()
    print('-----------------------Valid-----------------------')
    print('G Loss {:.7f}'.format(vGlosses.avg))
    print('D Loss {:.7f}'.format(vDlosses.avg))
    print('---------------------------------------------------')

    vGloss_plt.append(vGlosses.avg)
    vDloss_plt.append(vDlosses.avg)
    vGlosses.reset()
    vDlosses.reset()

    print()

    if ((j+1) % save_frq == 0 ) & (j != 0):
        plot((j+1), tGloss_plt, tDloss_plt, vGloss_plt, vDloss_plt)
        file_name = os.path.join(seed_dir, f'ckpts/modelG_epoch_{j+1}.pt')
        torch.save(model_G.state_dict(),file_name)
        file_name = os.path.join(seed_dir, f'ckpts/modelD_epoch_{j+1}.pt')
        torch.save(model_D.state_dict(),file_name)


# %%




