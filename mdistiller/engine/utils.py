import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller, tqdm_leave):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter), leave=tqdm_leave)

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg


def validate_npy(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        start_eval = True
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            output = nn.Softmax()(output)
            if start_eval:
                all_image = image.float().cpu()
                all_output = output.float().cpu()
                all_label = target.float().cpu()
                start_eval = False
            else:
                all_image = torch.cat((all_image, image.float().cpu()), dim=0)
                all_output = torch.cat((all_output, output.float().cpu()), dim=0)
                all_label = torch.cat((all_label, target.float().cpu()), dim=0)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    all_image, all_output, all_label = all_image.numpy(), all_output.numpy(), all_label.numpy()
    pbar.close()
    return top1.avg, top5.avg, losses.avg, all_image, all_output, all_label


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": "1;36",
        "TRAIN": "1;32",
        "EVAL": "1;31",
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def set_seed(seed=None):
    """Set the seed for reproducibility."""
    if seed is None:
        seed = torch.initial_seed() % (2**32)  # Generate a default seed if not provided
        print(f"Seed not provided. Using generated seed: {seed}")
    else:
        print(f"Using provided seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
    torch.backends.cudnn.deterministic = True  # Ensures reproducibility
    torch.backends.cudnn.benchmark = False
    return seed


def init_optimizer(model, cfg):
    if cfg.SOLVER.TYPE == "SGD":
        optimizer = optim.SGD(
            model.get_learnable_parameters(),
            lr=cfg.SOLVER.LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.SOLVER.TYPE} not implemented.")
    return optimizer

def init_scheduler(optimizer, cfg):
    if cfg.SOLVER.SCHEDULER == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.SOLVER.EPOCHS,
            eta_min=1e-5
        )
    elif cfg.SOLVER.SCHEDULER == "step":
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.SOLVER.LR_DECAY_STAGES,  # ex) [30, 60, 90]
            gamma=cfg.SOLVER.LR_DECAY_RATE  # ex) 0.1
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.SOLVER.SCHEDULER}")

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']



def setup_logger(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return SummaryWriter(os.path.join(log_path, "train.events"))

def log_training(writer, log_path, lr, epoch, log_dict, best_acc, wandb_enabled=False):
    # TensorBoard Logging
    for k, v in log_dict.items():
        writer.add_scalar(k, v, epoch)
    writer.flush()

    # Update best accuracy
    if log_dict["test_acc"] > best_acc:
        best_acc = log_dict["test_acc"]

    # WandB Logging
    if wandb_enabled:
        import wandb
        wandb.log({"current lr": lr})
        wandb.log(log_dict)
        if log_dict["test_acc"] > best_acc:
            wandb.run.summary["best_acc"] = best_acc

    # File Logging (worklog.txt)
    with open(os.path.join(log_path, "worklog.txt"), "a") as writer:
        lines = [
            "-" * 25 + os.linesep,
            f"epoch: {epoch}" + os.linesep,
            f"lr: {lr:.6f}" + os.linesep,
        ]
        for k, v in log_dict.items():
            lines.append(f"{k}: {v:.2f}" + os.linesep)
        lines.append("-" * 25 + os.linesep)
        writer.writelines(lines)

    return best_acc
