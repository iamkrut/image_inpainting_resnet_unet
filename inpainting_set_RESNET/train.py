import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from torchvision import transforms

import matplotlib.pyplot as plt

from model import ResNet
from dataloader import DataLoader
import random

import time

def train_net(net,
              epochs=100,
              data_dir='data/',
              lr=0.0007,
              save_cp=True,
              gpu=False):
    loader = DataLoader(data_dir)

    N_train = 100
 
    optimizer = optim.Adam(net.parameters(),
                            lr=lr,
                            weight_decay=0)
    criterion = nn.MSELoss()

    #print("Training start time: ", time.ctime())
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')

        epoch_loss = 0
        
        #composed = transforms.Compose([transforms.RandomRotation(60,), transforms.RandomCrop(224)])

        for i, (img, label) in enumerate(loader):
            
            print("index", i)

            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            img_tensor = torch.from_numpy(img).float()
            label_tensor = torch.from_numpy(label).float()

            img_tensor = img_tensor.permute(0,3,1,2) 
            label_tensor = label_tensor.permute(0,3,1,2)

            # todo: load image tensor to gpu
            if gpu:
                img_tensor = Variable(img_tensor.cuda())
                label_tensor = Variable(label_tensor.cuda())
            
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # todo: get prediction and getLoss()
            pred = net.forward(img_tensor)
            loss = criterion(pred, label_tensor)

            #loss = criterion(nn.Softmax(pred), label)
            epoch_loss += loss.item()
 
            print('Training sample %d / %d - Loss: %.6f' % (i+1, N_train, loss.item()))
            
            # optimize weights
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % (epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))
        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))

        images = img_tensor.cpu().detach().permute(0,2,3,1)
        labels = label_tensor.cpu().detach().permute(0,2,3,1)
        predictions = pred.cpu().detach().permute(0,2,3,1)

        index = random.randint(0, 15)

        image = str(epoch + 1) + "train_input_" + ".png"
        plt.imsave(join(data_dir, 'samples', image), images[index,:,:,:3])
        
        image = str(epoch + 1) + "train_groundtruth_" + ".png"
        plt.imsave(join(data_dir, 'samples', image), labels[index])
        
        image = str(epoch + 1) + "train_output_" + ".png"
        plt.imsave(join(data_dir, 'samples', image), predictions[index])

        #print("Training end time: ", time.ctime())
        # displays test images with original and predicted masks after training
        loader.setMode('test')
        net.eval()
        with torch.no_grad():
            for _, (img, label) in enumerate(loader):
                img_tensor = torch.from_numpy(img).float()
                label_tensor = torch.from_numpy(label).float()

                print(img_tensor.shape)
                print(label_tensor.shape)
                img_tensor = img_tensor.permute(0,3,1,2) 
                label_tensor = label_tensor.permute(0,3,1,2)

                # todo: load image tensor to gpu
                if gpu:
                    img_tensor = Variable(img_tensor.cuda())
                    label_tensor = Variable(label_tensor.cuda())
        
                # todo: get prediction
                pred = net.forward(img_tensor)

        images = img_tensor.cpu().detach().permute(0,2,3,1)
        labels = label_tensor.cpu().detach().permute(0,2,3,1)
        predictions = pred.cpu().detach().permute(0,2,3,1)

        index = random.randint(0, 15)

        image = str(epoch + 1) + "test_input_" + ".png"
        plt.imsave(join(data_dir, 'samples', image), images[index,:,:,:3])
        
        image = str(epoch + 1) + "test_groundtruth_" + ".png"
        plt.imsave(join(data_dir, 'samples', image), labels[index])
        
        image = str(epoch + 1) + "test_output_" + ".png"
        plt.imsave(join(data_dir, 'samples', image), predictions[index])

def getLoss(pred_label, target_label):
    loss = nn.MSELoss()
    return loss(pred_label, target_label)
# def getLoss(pred_label, target_label):
#     p = softmax(pred_label)
#     return cross_entropy(p, target_label)

# def softmax(input):
#     # todo: implement softmax function
#     exp_tensor = torch.exp(input)
#     sum_tensor = torch.sum(exp_tensor, dim = 1)
#     p = torch.div(exp_tensor, sum_tensor)
#     return p

# def cross_entropy(input, targets):
#     # todo: implement cross entropy
#     # Hint: use the choose function
#     ce = torch.mean(-torch.log(choose(input, targets)))
#     return ce

# # Workaround to use numpy.choose() with PyTorch
# def choose(pred_label, true_labels):
#     size = pred_label.size()
#     ind = np.empty([size[2]*size[3],3], dtype=int)
#     i = 0
#     for x in range(size[2]):
#         for y in range(size[3]):
#             ind[i,:] = [true_labels[x,y], x, y]
#             i += 1

#     pred = pred_label[0,ind[:,0],ind[:,1],ind[:,2]].view(size[2],size[3])

#     return pred
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = ResNet()

    if args.load:
        if args.gpu:
            net.load_state_dict(torch.load(args.load))
        else:
            net.load_state_dict(torch.load(args.load, map_location='cpu'))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        gpu=args.gpu,
        data_dir=args.data_dir)
