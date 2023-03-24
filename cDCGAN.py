from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import save_image
import time
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image

# Root directory for dataset
dataroot = 'C:/Users/Gordon/Desktop/project/testinput/img/'

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 50

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

num_classes = 46


# We can use an image folder dataset the way we have it setup.
# Create the dataset
eyes_color=[2,4,10,17,30,31,34,36,45]
hair_color=[5,6,11,18,19,29,32,37,40,44]
hair_length=[27,40,43]
ear_type=[1,12,16]
used= eyes_color+hair_color+hair_length+ear_type
def makeLabels():
    l=[]
    if random.randint(0,1)==1:
        idx_EC = eyes_color[random.randint(0,len(eyes_color)-1)]
    else:
        idx_EC=-1
    if random.randint(0, 1) == 1:
        idx_HC = hair_color[random.randint(0, len(hair_color)-1)]
    else:
        idx_HC=-1
    if random.randint(0, 1) == 1:
        idx_HL = hair_length[random.randint(0, len(hair_length)-1)]
    else:
        idx_HL=-1
    if random.randint(0, 1) == 1:
        idx_ET = ear_type[random.randint(0, len(ear_type)-1)]
    else:
        idx_ET=-1

    for i in range(num_classes):
        if i not in used:
            r = random.randint(0,3)
            if r==0:
                l.append(1)
            else:
                l.append(0)

        elif i==idx_EC or i==idx_HC or i==idx_HL or i==idx_ET:
            l.append(1)
        else:
            l.append(0)

    return l





class imgDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):

        self.landmarks_frame = pd.read_csv(csv_file)
        self.landmarks_frame['name'] = self.landmarks_frame['name'].astype(str)

        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = Image.open(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.flatten().astype('int')

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, ngf*ngf)

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs, labels):
        conditional = self.label_embedding(labels)
        # conditional = torch.flatten(conditional,1)
        conditional = torch.reshape(conditional, (batch_size, num_classes, ngf, ngf))

        conditional_inputs = torch.cat([inputs, conditional], dim=-3)
        out = self.main(conditional_inputs)
        out = out.reshape(out.size(0), nc+ngf, image_size, image_size)

        return out


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.label_embedding = nn.Embedding(num_classes, ndf*ndf)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc+num_classes, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs, labels):


        conditional = self.label_embedding(labels)
        #conditional = torch.flatten(conditional,1)
        conditional = torch.reshape(conditional,(batch_size,num_classes,ndf,ndf))

        conditional_inputs = torch.cat([inputs, conditional], dim=-3)
        out = self.main(conditional_inputs)

        return out


if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 42
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    os.makedirs("images", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    face_dataset = imgDataset(csv_file='testinput/labels.csv',
                              root_dir=dataroot, transform=transform)
    labelsTag = face_dataset.landmarks_frame.columns.values.tolist()[1:]

    # print(sample)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(face_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    #for i in range(len(face_dataset)):
    #    sample = face_dataset[i]
    #    print(i,sample['image'].shape,sample['landmarks'].shape)
    #    if i==3:
    #        break
    #for i,sample in enumerate(dataloader):
    #    print(i,sample)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    # Generator Code

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    t=makeLabels()
    tx=np.asarray(t).reshape(num_classes)
    arr = np.zeros((64,num_classes))
    arr[...]=tx

    result = [x for x, y in zip(labelsTag, t) if y == 1]
    # random seed=42 bangs blush braids chocker earrings hair_ornament maid headdress silver hair
    fixed_conditional = torch.as_tensor(arr,device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    # For each epoch
    tic = time.perf_counter()
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        # for i, (data, _) in enumerate(dataloader):
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            inputs=data['image'].to(device)
            labels=data['landmarks'].to(device)

            b_size = inputs.size(0)

            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D

            output = netD(inputs,labels).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            arr = np.zeros((b_size, num_classes))
            for i in range(b_size):
                t = makeLabels()
                arr[i] = t
            arr=arr.astype(int)
            conditional = torch.as_tensor(arr,device=device)

            # Generate fake image batch with G
            fake = netG(noise,conditional)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake,conditional).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                print(time.perf_counter() - tic)
                tic = time.perf_counter()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                save_image(vutils.make_grid(fake, padding=2, normalize=True), "images/%d.png" % iters)

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    torch.save(netG.state_dict(), 'generator')
    torch.save(netD.state_dict(), 'discriminator')
