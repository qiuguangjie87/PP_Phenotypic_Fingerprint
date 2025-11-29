import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from nets.LGASSNet import LGASSNet
from nets.LGASSNet_training import (CE_Loss, Dice_loss, Focal_Loss, get_lr_scheduler, set_optimizer_lr)
from utils.dataloader import LGASSNet_Dataset, LGASSNet_dataset_collate
from utils.utils_metrics import f_score


def train_model():
    Cuda = True
    num_classes = 3 + 1
    backbone = "lwganet_l0"   # "lwganet_l0", "lwganet_l1", "lwganet_l2"
    input_shape = [512, 512]
    downsample_factor = 16

    Init_Epoch = 0
    Freeze_Epoch = 50
    UnFreeze_Epoch = 400
    Freeze_batch_size = 16
    Unfreeze_batch_size = 16
    Freeze_Train = True

    Init_lr = 5e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.937
    weight_decay = 0
    lr_decay_type = 'cos'

    data_path = 'data'

    dice_loss = True
    focal_loss = True
    cls_weights = np.ones([num_classes], np.float32)

    if Cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        Cuda = False

    model = LGASSNet(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=True)

    if Cuda:
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.to(device)

    with open(os.path.join(data_path, "Lettuce/ImageSets/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(data_path, "Lettuce/ImageSets/val.txt"), "r") as f:
        val_lines = f.readlines()

    print(f"Train samples: {len(train_lines)}, Val samples: {len(val_lines)}")

    def fit_one_epoch(model_train, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, optimizer):
        model_train.train()
        total_loss = 0
        total_f_score = 0

        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            if Cuda:
                imgs = imgs.cuda(non_blocking=True)
                pngs = pngs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            else:
                imgs = imgs.to(device)
                pngs = pngs.to(device)
                labels = labels.to(device)

            weights = torch.from_numpy(cls_weights)
            if Cuda:
                weights = weights.cuda(non_blocking=True)
            else:
                weights = weights.to(device)

            optimizer.zero_grad()
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            if iteration % 10 == 0:
                print(f'Epoch: {epoch + 1}/{Epoch}, Step: {iteration}/{epoch_step}, '
                      f'Loss: {total_loss / (iteration + 1):.4f}, F-score: {total_f_score / (iteration + 1):.4f}, '
                      f'Device: {imgs.device}')

        model_train.eval()
        val_loss = 0
        val_f_score = 0

        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                if Cuda:
                    imgs = imgs.cuda(non_blocking=True)
                    pngs = pngs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                else:
                    imgs = imgs.to(device)
                    pngs = pngs.to(device)
                    labels = labels.to(device)

                weights = torch.from_numpy(cls_weights)
                if Cuda:
                    weights = weights.cuda(non_blocking=True)
                else:
                    weights = weights.to(device)

                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                _f_score = f_score(outputs, labels)
                val_loss += loss.item()
                val_f_score += _f_score.item()

        print(f'Epoch: {epoch + 1}/{Epoch} Completed')
        print(f'Train Loss: {total_loss / epoch_step:.4f}, Train F-score: {total_f_score / epoch_step:.4f}')
        print(f'Val Loss: {val_loss / epoch_step_val:.4f}, Val F-score: {val_f_score / epoch_step_val:.4f}')

        if (epoch + 1) % 5 == 0:
            save_path = f'logs/ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f'Model saved: {save_path}')

    model_train = model.train()
    UnFreeze_flag = False

    if Freeze_Train:
        if isinstance(model, torch.nn.DataParallel):
            for param in model.module.backbone.parameters():
                param.requires_grad = False
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
        print("Backbone frozen for initial training")

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            print("Unfreezing backbone...")
            if isinstance(model, torch.nn.DataParallel):
                for param in model.module.backbone.parameters():
                    param.requires_grad = True
            else:
                for param in model.backbone.parameters():
                    param.requires_grad = True
            batch_size = Unfreeze_batch_size
            UnFreeze_flag = True
        else:
            batch_size = Freeze_batch_size if Freeze_Train and not UnFreeze_flag else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            print("Warning: Batch size too large, reducing batch size")
            batch_size = min(batch_size, len(train_lines), len(val_lines))
            epoch_step = len(train_lines) // batch_size
            epoch_step_val = len(val_lines) // batch_size

        train_dataset = LGASSNet_Dataset(train_lines, input_shape, num_classes, True, data_path)
        val_dataset = LGASSNet_Dataset(val_lines, input_shape, num_classes, False, data_path)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                         pin_memory=True, drop_last=True, collate_fn=LGASSNet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4,
                             pin_memory=True, drop_last=True, collate_fn=LGASSNet_dataset_collate)

        print(f'Epoch {epoch + 1}/{UnFreeze_Epoch}, Batch size: {batch_size}, LR: {Init_lr_fit:.6f}')


        fit_one_epoch(model_train, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, optimizer)

    print("Training completed!")


if __name__ == "__main__":

    if not os.path.exists('logs'):
        os.makedirs('logs')

    train_model()