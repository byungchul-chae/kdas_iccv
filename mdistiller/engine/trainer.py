import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import OrderedDict
import getpass
from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
    init_optimizer,
    init_scheduler,
    get_lr,
    setup_logger,
    log_training
)
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = init_optimizer(self.distiller.module, cfg)
        self.scheduler = init_scheduler(self.optimizer, cfg)
        self.best_acc = -1
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        self.tf_writer = setup_logger(self.log_path)
        self.tqdm_leave = cfg.LOG.TQDM_LEAVE

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_acc = state["best_acc"]
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best accuracy:{:.2f}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
            "batch_size": AverageMeter()
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter), leave=self.tqdm_leave)

        # train loops
        self.distiller.train()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()

        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller, self.tqdm_leave)

        if self.scheduler:
            self.scheduler.step()
        lr = get_lr(self.optimizer)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss
            }
        )
        self.best_acc = log_training(self.tf_writer, self.log_path, lr, epoch, log_dict, self.best_acc, self.cfg.LOG.WANDB)

        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.mean().cpu().detach().item(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class AugTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image_weak, image_strong = image
        image_weak, image_strong = image_weak.float(), image_strong.float()
        image_weak, image_strong = image_weak.cuda(non_blocking=True), image_strong.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(image_weak=image_weak, image_strong=image_strong, target=target, epoch=epoch)

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image_weak.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class KDASTrainer(BaseTrainer):
    """
    KDASTrainer (Batch-level Sampling with Dynamic Regularization)

    [Functionality]
      - During warmup epochs, performs standard Knowledge Distillation.
      - After warmup:
        - In sampling epochs, selects samples based on teacher–student KL divergence (kl_ts) using logits from both teacher and student on an enlarged batch.
        - In non-sampling epochs, trains using cached indices from the previous sampling epoch.
        - The sampling ratio changes smoothly per epoch via a cosine annealing schedule.

    [Required cfg fields]
      - cfg.TRAIN.BATCH_SIZE: Effective batch size for final training
      - cfg.KDAS.START_RATE, cfg.KDAS.END_RATE: Sample selection ratio per epoch (e.g., 0.6 → 0.4)
      - cfg.SOLVER.EPOCHS: Total number of epochs
      - cfg.KDAS.SAMPLING_PERIOD: Sampling epoch period (int or list)
      - cfg.KDAS.EXCLUSION_RATE: Outlier sample exclusion ratio (e.g., 0.05)
      - cfg.KDAS.WARMUP_EPOCHS: Number of warmup epochs
      - cfg.SOLVER.LR: Learning rate, etc.
    """

    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg):
        super(KDASTrainer, self).__init__(experiment_name, distiller, train_loader, val_loader, cfg)
        # Effective batch size for final training
        self.target_batch_size = cfg.SOLVER.BATCH_SIZE
        # Sample selection ratio at the start and end epoch
        self.start_rate = cfg.KDAS.START_RATE
        self.end_rate = cfg.KDAS.END_RATE
        self.total_epochs = cfg.SOLVER.EPOCHS
        # Sampling period (int or list)
        self.sampling_period_config = cfg.KDAS.SAMPLING_PERIOD
        self.exclusion_rate = cfg.KDAS.EXCLUSION_RATE

        # Warmup configuration
        self.warmup_epochs = cfg.KDAS.WARMUP_EPOCHS
        self.is_warmup = True  # Initially in warmup mode

        # Temporarily store selected indices for each sampling epoch
        self.epoch_cached_indices = []
        # Cached indices from the previous sampling epoch (used in non-sampling epochs)
        self.cached_indices = None

        # Flag indicating whether this is a sampling epoch
        self.do_sample = False

        # Original dataset (stored for DataLoader re-creation)
        self.train_dataset = self.train_loader.dataset

        # Progress tracking (sample counter, etc.)
        self.sample_counter = 0

    def is_in_warmup(self, epoch):
        """
        Check if the current epoch is within the warmup period.
        """
        return epoch <= self.warmup_epochs

    def get_current_rate(self, epoch):
        """
        Calculate the sample selection ratio for the current epoch.
        During warmup, use 1.0 (use all data).
        After warmup, apply cosine annealing.
        """
        if self.is_in_warmup(epoch):
            return 1.0

        # Calculate progress after warmup
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        rate = self.end_rate + 0.5 * (self.start_rate - self.end_rate) * (
                1 + math.cos(math.pi * (adjusted_epoch - 1) / (adjusted_total - 1))
        )
        return rate

    def get_current_sampling_period(self, epoch):
        """
        If cfg.KDAS.SAMPLING_PERIOD is an integer, use it as is.
        If it is a list, divide the total epochs into segments and return the period for the current segment.
        Calculation is based on epochs after warmup.
        """
        if isinstance(self.sampling_period_config, int):
            return self.sampling_period_config

        adjusted_epoch = epoch - self.warmup_epochs
        periods = self.sampling_period_config
        num_segments = len(periods)
        segment_length = (self.total_epochs - self.warmup_epochs) // num_segments
        segment_index = (adjusted_epoch - 1) // segment_length

        if segment_index >= num_segments:
            segment_index = num_segments - 1
        return periods[segment_index]

    def is_sampling_epoch(self, epoch):
        """
        Determine whether sampling should be performed in the current epoch.
        No sampling during warmup period.
        """
        if self.is_in_warmup(epoch):
            return False

        adjusted_epoch = epoch - self.warmup_epochs
        period = self.get_current_sampling_period(epoch)
        return (adjusted_epoch - 1) % period == 0

    def update_train_loader(self, epoch):
        """
        Reconstruct the DataLoader based on the current epoch.
          - Warmup epoch: Use the entire dataset with the default batch size.
          - Sampling epoch: Use the entire dataset with an enlarged batch size (shuffled).
          - Non-sampling epoch: Use cached indices from the previous sampling epoch.
        """
        if self.is_in_warmup(epoch):
            self.do_sample = False
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.target_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            return

        current_rate = self.get_current_rate(epoch)
        if self.is_sampling_epoch(epoch):
            self.do_sample = True
            self.epoch_cached_indices = []
            # Enlarged batch size: target_batch_size / current_rate (round up)
            enlarged_bs = math.ceil(self.target_batch_size / current_rate)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=enlarged_bs,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
        else:
            self.do_sample = False
            if self.cached_indices is not None and len(self.cached_indices) > 0:
                new_sampler = SubsetRandomSampler(self.cached_indices)
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.target_batch_size,
                    sampler=new_sampler,
                    num_workers=2,
                    pin_memory=True,
                )
            else:
                print(f"Epoch {epoch}: No cached indices found. Using default DataLoader.")

    def train_epoch(self, epoch):
        self.update_train_loader(epoch)
        self.sample_counter = 0
        ret = super(KDASTrainer, self).train_epoch(epoch)
        if self.do_sample:
            # At the end of the epoch, remove duplicates and cache (order is not important)
            unique_indices = list(set(self.epoch_cached_indices))
            self.cached_indices = unique_indices
        return ret

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()

        # Unpack data
        images, targets, indices = data
        images = images.float().cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        indices = indices.cuda(non_blocking=True)

        if self.is_in_warmup(epoch):
            # During warmup: perform standard Knowledge Distillation
            preds, losses_dict = self.distiller(image=images, target=targets, epoch=epoch)
            current_images = images
            current_targets = targets
        elif self.do_sample:
            # Apply KDAS sampling using KL divergence
            student_logits, teacher_logits = self.distiller.module.get_logits(images)
            temperature = 1

            # Compute KL divergence between teacher and student
            log_pred_student = F.log_softmax(student_logits / temperature, dim=1)
            pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
            kl_ts = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(dim=1) * (temperature ** 2)

            # Select effective batch based on KL divergence
            effective_bs = self.target_batch_size
            N = images.size(0)
            if N < effective_bs:
                selection = torch.arange(N).to(images.device)
            else:
                sorted_metric, sorted_indices = torch.sort(kl_ts, descending=True)
                # Calculate exclusion rate for the current epoch (linear decay after epoch 180)
                if epoch <= 150:
                    current_exclusion_rate = self.exclusion_rate
                else:
                    # At epoch 180, exclusion_rate; at the final epoch, decay to 0 linearly
                    current_exclusion_rate = self.exclusion_rate * (self.total_epochs - epoch) / (self.total_epochs - 150)
                    current_exclusion_rate = max(current_exclusion_rate, 0)
                skip_num = int(math.ceil(N * current_exclusion_rate))
                candidate_indices = sorted_indices[skip_num:]
                selection = candidate_indices[:effective_bs]

            # Cache selected indices
            selected_dataset_indices = indices[selection].detach().cpu().tolist()
            self.epoch_cached_indices.extend(selected_dataset_indices)

            # Extract selected samples
            current_images = images[selection]
            current_targets = targets[selection]

            # Forward and compute loss for selected samples
            preds, losses_dict = self.distiller(image=current_images, target=current_targets, epoch=epoch)

        else:
            # Non-sampling epoch
            preds, losses_dict = self.distiller(image=images, target=targets, epoch=epoch)
            current_images = images
            current_targets = targets

        # Backward and optimizer step
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()

        # Update metrics
        batch_size = current_images.size(0)
        acc1, acc5 = accuracy(preds, current_targets, topk=(1, 5))

        train_meters["losses"].update(loss.item(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        train_meters["batch_size"].update(batch_size)

        phase = "Warmup" if self.is_in_warmup(epoch) else "KDAS"
        msg = (f"Epoch:{epoch} ({phase}) | Samples:{self.sample_counter} | "
               f"Batch: {train_meters['batch_size'].avg:.3f} | "
               f"Loss:{train_meters['losses'].avg:.4f} | "
               f"Top-1:{train_meters['top1'].avg:.3f} | "
               f"Top-5:{train_meters['top5'].avg:.3f}")

        self.sample_counter += batch_size
        return msg
        