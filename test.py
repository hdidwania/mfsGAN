import argparse
import os

from data import FGDataset
from model import Generator

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score
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

    # Take the later set of images as train set
    train_ratio = 0.8
    nb_samples = len(data_files)
    nb_train = int(train_ratio * nb_samples)
    print("Number of training samples:", nb_train)

    train_files = data_files[:nb_train]
    test_files = data_files[nb_train:]
    nb_test = len(test_files)
    print("Number of testing samples:", nb_test)
    
    train_dataset = FGDataset(train_files, ip_data_path, gt_data_path, n=args.n)
    test_dataset = FGDataset(test_files, ip_data_path, gt_data_path, n=args.n)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

    # Loading the trained generator
    gen = Generator(n=args.n).double().cuda()
    gen.load_state_dict(torch.load(os.path.join(args.model_path, "gen.pt")))
    gen.eval()

    # Structures to store metrics
    train_metrics = {
        "precision": list(),
        "recall": list(),
        "f1": list()
    }
    test_metrics = {
        "precision": list(),
        "recall": list(),
        "f1": list()
    }

    # Calculate metrics on train set
    for step, (ip_data, target_data) in enumerate(train_dataloader):
        print("\rImage {}".format(step), end="")
        for i, data_sample in enumerate(ip_data):
            ip_data[i] = data_sample.double().cuda()
        
        fg_segmentation, _, _, _ = gen(ip_data)
        
        fg_segmentation = np.squeeze(np.array(fg_segmentation.cpu().detach()))
        fg_segmentation = np.where(fg_segmentation < 0.5, 0, 1).astype(np.int)
        gt_segmentation = np.squeeze(np.array(target_data[0].cpu().detach())).astype(np.int)
        
        p = precision_score(gt_segmentation.reshape(-1,), fg_segmentation.reshape(-1,))
        r = recall_score(gt_segmentation.reshape(-1,), fg_segmentation.reshape(-1,))
        f = 2*p*r / (p + r)
        train_metrics["precision"].append(p)
        train_metrics["recall"].append(r)
        train_metrics["f1"].append(f)
    print()
    print("Train Set Metrics:")
    print("Precision:", np.mean(np.array(train_metrics['precision'])))
    print("Recall:", np.mean(np.array(train_metrics['recall'])))
    print("F-Measure:", np.mean(np.array(train_metrics['f1'])))

    # Calculate metrics on test set
    for step, (ip_data, target_data) in enumerate(test_dataloader):
        print("\rImage {}".format(step), end="")
        for i, data_sample in enumerate(ip_data):
            ip_data[i] = data_sample.double().cuda()
        
        fg_segmentation, _, _, _ = gen(ip_data)
        
        fg_segmentation = np.squeeze(np.array(fg_segmentation.cpu().detach()))
        fg_segmentation = np.where(fg_segmentation < 0.5, 0, 1).astype(np.int)
        gt_segmentation = np.squeeze(np.array(target_data[0].cpu().detach())).astype(np.int)
        
        p = precision_score(gt_segmentation.reshape(-1,), fg_segmentation.reshape(-1,))
        r = recall_score(gt_segmentation.reshape(-1,), fg_segmentation.reshape(-1,))
        f = 2*p*r / (p + r)
        test_metrics["precision"].append(p)
        test_metrics["recall"].append(r)
        test_metrics["f1"].append(f)
    print()
    print("Test Set Metrics:")
    print("Precision:", np.mean(np.array(test_metrics['precision'])))
    print("Recall:", np.mean(np.array(test_metrics['recall'])))
    print("F-Measure:", np.mean(np.array(test_metrics['f1'])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    main(args)