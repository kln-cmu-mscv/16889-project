import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import wandb

from network import PoseRegressor_sincos
from utils import *
from pose_dataset import PoseDataset

parser = argparse.ArgumentParser(description='Pose Regressor Testing')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '-b',
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')

parser.add_argument(
    '-w',
    '--wandb',
    dest='wandb',
    action='store_true',
    help='wandb logs')


def main():
    args = parser.parse_args()

    if args.wandb:
      wandb.init(project="3d-project")

    checkpoint = torch.load('checkpoint.pt')
    
    # create model
    print("Creating Pose Regressor model")
    model = PoseRegressor_sincos(pretrained = True)
    print(model)

    model = model.cuda()
    model.load_state_dict(checkpoint)

    criterion_recon= nn.MSELoss()
    criterion_MAE = nn.L1Loss()
    
    val_dataset = PoseDataset(split = 'test', data_dir = '16889_pose_dataset_chair_highres')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda x: x)

    print("Testing trained Pose Regressor model")
    
    validate(val_loader, model, criterion_recon, criterion_MAE, args.batch_size, args.wandb)

            
    
def validate(val_loader, model, criterion_recon, criterion_MAE, batch_size, USE_WANDB=False):
        
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_recon = AverageMeter()
    losses_MAE_theta = AverageMeter()
    losses_MAE_phi = AverageMeter()
    losses_constraint = AverageMeter()
    losses = AverageMeter()
    
    model.eval()

    accuracy_phi_2 = 0
    accuracy_phi_5 = 0
    accuracy_phi_10 = 0
    accuracy_theta_2 = 0
    accuracy_theta_5 = 0
    accuracy_theta_10 = 0

    end = time.time()
    for i, (data) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = [d['image'].unsqueeze(0) for d in data]
        inputs = torch.cat(inputs, dim = 0).to('cuda')

        og_inputs = [d['og_image'].unsqueeze(0) for d in data]
        og_inputs = torch.cat(og_inputs, dim = 0).to('cuda')

        target_theta = [d['theta'].unsqueeze(0) for d in data]
        target_theta = torch.cat(target_theta, dim = 0).unsqueeze(1).to('cuda')

        target_phi = [d['phi'].unsqueeze(0) for d in data]
        target_phi = torch.cat(target_phi, dim = 0).unsqueeze(1).to('cuda')
        
        output = model(inputs)

        pred_phi_sin = output[0]
        pred_phi_cos = output[1]
        pred_theta_sin = output[2]
        pred_theta_cos = output[3]
        
        pred_theta = (torch.atan2(pred_theta_sin, pred_theta_cos) * 180)/(2*np.pi) 
        pred_phi = (torch.atan2(pred_phi_sin, pred_phi_cos) * 180)/np.pi

        rendered_img = render_image(pred_theta, pred_phi)

        loss_recon = criterion_recon(rendered_img, og_inputs)
        loss_constraint = constraint(output)
        loss_MAE_phi = criterion_MAE(pred_phi, target_phi)
        loss_MAE_theta = criterion_MAE(pred_theta, target_theta)
        loss = loss_recon + loss_constraint + 0.1 * loss_MAE_theta + 0.1 * loss_MAE_phi

        # measure metrics and record loss
        losses.update(loss.item(), inputs.size(0))
        losses_recon.update(loss_recon.item(), inputs.size(0))
        losses_constraint.update(loss_constraint.item(), inputs.size(0))
        losses_MAE_phi.update(loss_MAE_phi.item(), inputs.size(0))
        losses_MAE_theta.update(loss_MAE_theta.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        accuracy_phi_2 += torch.sum(torch.abs(target_phi - pred_phi) < 2, dim = 0).item()
        accuracy_phi_5 += torch.sum(torch.abs(target_phi - pred_phi) < 5, dim = 0).item()
        accuracy_phi_10 += torch.sum(torch.abs(target_phi - pred_phi) < 10, dim = 0).item()
        accuracy_theta_2 += torch.sum(torch.abs(target_theta - pred_theta) < 2, dim = 0).item()
        accuracy_theta_5 += torch.sum(torch.abs(target_theta - pred_theta) < 5, dim = 0).item()
        accuracy_theta_10 += torch.sum(torch.abs(target_theta - pred_theta) < 10, dim = 0).item()

        print("predicted Phi min: {}   predicted Phi max: {}".format(torch.min(torch.abs(target_phi - pred_phi)).item(), 
                                           torch.max(torch.abs(target_phi - pred_phi)).item())   )  

        print("predicted Theta min: {}   predicted Theta max: {}".format(torch.min(torch.abs(target_theta - pred_theta)).item(), 
                                           torch.max(torch.abs(target_theta - pred_theta)).item())   )  

        print('Epoch: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss Recon {loss_recon.val:.9f} ({loss_recon.avg:.4f})\t'
                'Loss Constraint {loss_constraint.val:.9f} ({loss_constraint.avg:.4f})\t'
                'Loss MAE Theta {loss_MAE_theta.val:.9f} ({loss_MAE_theta.avg:.4f})\t'
                'Loss MAE Phi {loss_MAE_phi.val:.9f} ({loss_MAE_phi.avg:.4f})\t'
                'Loss {loss.val:.9f} ({loss.avg:.4f})\t'.format(
                
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss_recon=losses_recon,
                    loss_constraint=losses_constraint,
                    loss_MAE_theta=losses_MAE_theta,
                    loss_MAE_phi=losses_MAE_phi, 
                    loss= losses
                    ))
        
        # Loss curves
        if USE_WANDB:
              wandb.log({ 
                        'iterations':  i, 
                        'loss ': losses.val})

              wandb.log({ 
                        'iterations':  i, 
                        'loss Recon ': losses_recon.val})

              wandb.log({
                        'iterations':  i, 
                        'loss MAE Phi': losses_MAE_phi.val})

              wandb.log({ 
                        'iterations':  i, 
                        'loss MAE Theta': losses_MAE_theta.val})

              wandb.log({
                        'iterations':  i, 
                        'loss constraint': losses_constraint.val})

        # Images
        if USE_WANDB:
            if i % 20 == 0:
                for j in range(0, len(og_inputs), 5):
                    img1 = wandb.Image(rendered_img[j], caption="Rendered Image Azim: {} Elev: {}".format(pred_phi[j].item(), pred_theta[j].item()))
                    img2 = wandb.Image(og_inputs[j], caption="Ground Truth Image Azim:{} Elev: {}".format(target_phi[j].item(), target_theta[j].item()))
                    wandb.log({'GT/pred images:' : [img2, img1]})
                
                for j in range(len(og_inputs)):
                    if abs(target_phi[j] - pred_phi[j]) > 10 or abs(target_theta[j] - pred_theta[j]) > 10:
                        img1 = wandb.Image(rendered_img[j], caption="Rendered Image Azim: {} Elev: {}".format(pred_phi[j].item(), pred_theta[j].item()))
                        img2 = wandb.Image(og_inputs[j], caption="Ground Truth Image Azim:{} Elev: {}".format(target_phi[j].item(), target_theta[j].item()))
                        wandb.log({'Bad Predictions GT/pred images:' : [img2, img1]})

    
    accuracy_phi_2 = accuracy_phi_2/(len(val_loader) * batch_size)
    accuracy_phi_5 = accuracy_phi_5/(len(val_loader) * batch_size)
    accuracy_phi_10 = accuracy_phi_10/(len(val_loader) * batch_size)
    accuracy_theta_2 = accuracy_theta_2/(len(val_loader) * batch_size)
    accuracy_theta_5 = accuracy_theta_5/(len(val_loader) * batch_size)
    accuracy_theta_10 = accuracy_theta_10/(len(val_loader) * batch_size)
    
    print(" Threshold 2 degrees Theta Accuracy: {}  Phi Accuracy: {}".format(accuracy_theta_2, accuracy_phi_2))
    print(" Threshold 5 degrees Theta Accuracy: {}  Phi Accuracy: {}".format(accuracy_theta_5, accuracy_phi_5))
    print(" Threshold 10 degrees Theta Accuracy: {}  Phi Accuracy: {}".format(accuracy_theta_10, accuracy_phi_10))

    # Accuracies
    if USE_WANDB:
        wandb.log({
                'Accuracy_theta_thresh=2': accuracy_theta_2,
                'Accuracy_theta_thresh=5': accuracy_theta_5,
                'Accuracy_theta_thresh=10': accuracy_theta_10, 
                'Accuracy_phi_thresh=2': accuracy_phi_2,
                'Accuracy_phi_thresh=5': accuracy_phi_5,
                'Accuracy_phi_thresh=10': accuracy_phi_10 })


if __name__ == '__main__':
    main()