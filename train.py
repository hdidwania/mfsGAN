import argparse
import os
import time

from data import FGDataset
from model import Generator, Discriminator

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def get_target_frame(ip, n):
    l = n//2 * 3
    r = l + 3
    return ip[:, l:r]


def main(args):
    # Data path contains following folder and files- input, groundtruth, temoralROI.txt
    # input and groundtruth contain images and corresponding data
    ip_data_path = os.path.join(args.data_path, 'input')
    gt_data_path = os.path.join(args.data_path, 'groundtruth')
    nb_samples_f = os.path.join(args.data_path, 'temporalROI.txt')

    data_files = list(zip(sorted(os.listdir(ip_data_path), key=lambda x: int(x[2:-4])),
                     sorted(os.listdir(gt_data_path), key=lambda x: int(x[2:-4])))) 
    with open(nb_samples_f, 'r') as f:
        nb_no_gt, _ = map(int, f.read().split(' '))
    data_files = data_files[nb_no_gt:]

    # Take the first set of images as train set
    train_ratio = 0.8
    nb_samples = len(data_files)
    nb_train = int(train_ratio * nb_samples)
    print("Number of training samples:", nb_train)
    train_files = data_files[:nb_train]
    
    train_dataset = FGDataset(train_files, ip_data_path, gt_data_path, n=args.n)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
    
    # Loading the models
    gen = Generator(n=args.n).double().cuda()
    dis = Discriminator().double().cuda()
    
    # Setting loss functions and optimizer
    loss_fn = nn.BCELoss().cuda()

    optimizer_G = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(dis.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Tensors corresponding to real and fake labels 
    real_label = torch.from_numpy(np.ones((args.batchsize, 1, 15, 15))).cuda()
    fake_label = torch.from_numpy(np.zeros((args.batchsize, 1, 15, 15))).cuda()

    generator_loss_values = list()
    discriminator_loss_values = list()

    start_epoch = 0

    if args.resume_from:
        start_epoch = args.resume_from

        gen.load_state_dict(torch.load(os.path.join(args.output_dir, str(start_epoch), "gen.pt")))
        dis.load_state_dict(torch.load(os.path.join(args.output_dir, str(start_epoch), "dis.pt")))

        loss_value_dict = np.load(os.path.join(args.output_dir, str(start_epoch), "loss_values.npy"), allow_pickle=True).item()
        generator_loss_values = loss_value_dict['generator']
        discriminator_loss_values = loss_value_dict['discriminator']

    for epoch in range(start_epoch, start_epoch+args.nb_epochs):
        os.makedirs(os.path.join(args.output_dir, str(epoch+1)), exist_ok=True)
        epoch_start_time = time.time()
        for step, (ip_data, target_data) in enumerate(train_dataloader):
            for i, data_sample in enumerate(ip_data):
                ip_data[i] = data_sample.double().cuda()
            for i, data_sample in enumerate(target_data):
                target_data[i] = data_sample.double().cuda()

            gt_1, gt_2, gt_3 = target_data
            real_seg = gt_1
            target_frame_in = get_target_frame(ip_data[0], args.n)

            # Training generator
            optimizer_G.zero_grad()

            fake_seg_gen, recons_feature_1, recons_feature_2, recons_feature_3 = gen(ip_data)
            fake_in_gen = torch.cat([fake_seg_gen, target_frame_in], dim=1)
            pred_fake_gen = dis(fake_in_gen)
            
            loss_G_gan = loss_fn(pred_fake_gen, real_label)
            loss_G_recons_full = loss_fn(fake_seg_gen, real_seg)
            loss_G_recons_1 = loss_fn(recons_feature_1, gt_1)
            loss_G_recons_2 = loss_fn(recons_feature_2, gt_2)
            loss_G_recons_3 = loss_fn(recons_feature_3, gt_3)
            loss_G_recons_feature = loss_G_recons_1 + loss_G_recons_2 + loss_G_recons_3
            loss_G = loss_G_gan + 10*loss_G_recons_full + 2*loss_G_recons_feature
            loss_G.backward()
            optimizer_G.step()

            # Training discriminator
            optimizer_D.zero_grad()

            fake_seg_dis, _, _, _ = gen(ip_data)
            fake_in_dis = torch.cat([fake_seg_dis, target_frame_in], dim=1)
            pred_fake_dis = dis(fake_in_dis)
            loss_D_fake = loss_fn(pred_fake_dis, fake_label)

            real_in_dis = torch.cat([real_seg, target_frame_in], dim=1)
            pred_real_dis = dis(real_in_dis)
            loss_D_real = loss_fn(pred_real_dis, real_label)

            loss_D = 0.5 * (loss_D_fake + loss_D_real)
            loss_D.backward()
            optimizer_D.step()

            print("\rEpoch {} Step {} [Generator Loss Total {:.3f} GAN {:.3f} Recons {:.3f}] [Discriminator Loss {:.3f}] Epoch Time {:.0f}s ".format(
                epoch+1, step, loss_G, loss_G_gan, loss_G_recons_full+loss_G_recons_feature, loss_D, time.time()-epoch_start_time), end="")

            # Saving loss values
            if step % 10 == 0:
                generator_loss_values.append(loss_G.item())
                discriminator_loss_values.append(loss_D.item())
                
            # Plotting sample images
            if step % 100 == 0:
                target_frame_in = np.squeeze(np.array(target_frame_in.cpu().detach()))
                target_frame_in = np.moveaxis(target_frame_in, 0, -1)
                target_frame_in = (target_frame_in + 1) / 2 * 255
                target_frame_in = target_frame_in.astype(int)
                
                fake_seg = np.squeeze(np.array(fake_seg_dis.cpu().detach()))
                real_seg = np.squeeze(np.array(real_seg.cpu().detach()))
                
                plt.figure(figsize=(10,5))
                plt.subplot(1, 3, 1)
                plt.imshow(target_frame_in)
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(fake_seg)
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(real_seg)
                plt.axis('off')
                plt.savefig(os.path.join(args.output_dir, str(epoch+1), "step_{}.jpg".format(step)))
                plt.close()
        print()
        
        # Saving the model
        torch.save(gen.state_dict(), os.path.join(args.output_dir, str(epoch+1), "gen.pt"))
        torch.save(dis.state_dict(), os.path.join(args.output_dir, str(epoch+1), "dis.pt"))
        np.save(os.path.join(args.output_dir, str(epoch+1), "loss_values.npy"), np.array(
            {
                'generator': generator_loss_values,
                'discriminator': discriminator_loss_values
            }
        ))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--nb_epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()

    main()