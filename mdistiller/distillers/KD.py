import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller

def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)

def kd_loss(logits_student_in, logits_teacher_in, temperature, logit_stand, reduction="mean"):
    """
    Compute the KD loss between teacher and student outputs.
    If reduction == "none", returns a tensor of shape [batch_size] (per-sample loss).
    Otherwise, returns a scalar.
    """
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in
    pred_student = F.softmax(logits_student / temperature, dim=1)
    log_pred_teacher = F.log_softmax(logits_teacher / temperature, dim=1)
    loss = F.kl_div(log_pred_teacher, pred_student, reduction="none").sum(dim=1)
    loss = loss * (temperature ** 2)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Unknown reduction: {}".format(reduction))


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network.

    Warmup period:
    - 일반적인 KD loss 계산 (penalty 없음)
    - CE loss와 KD loss의 기본 가중치 사용

    Post-warmup period:
    - KDAS trainer 사용 시 penalty reweighting 적용
    - teacher-target divergence 기반 동적 가중치 조정
    """

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.cfg = cfg
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.logit_stand = cfg.EXPERIMENT.LOGIT_STAND

        # Warmup
        self.warmup_epochs = cfg.KDAS.WARMUP_EPOCHS if hasattr(cfg.KDAS, 'WARMUP_EPOCHS') else 0

        # Activate penalty mode if using "kdas"
        if cfg.SOLVER.TRAINER == "kdas":
            self.penalty_factor = cfg.KDAS.PENALTY_FACTOR
            self.penalty_lambda = cfg.KDAS.PENALTY_LAMBDA
            # penalty warmup은 KDAS warmup 이후부터 시작
            self.penalty_warmup = cfg.KDAS.PENALTY_WARMUP
        else:
            self.penalty_factor = None
            self.penalty_lambda = None
            self.penalty_warmup = None

    def is_in_warmup(self, epoch):
        """현재 epoch가 warmup 기간인지 확인"""
        return epoch <= self.warmup_epochs

    def get_penalty_progress(self, epoch):
        """
        Penalty 적용 진행도 계산
        - Warmup 기간: 0
        - Warmup 이후: penalty_warmup에 따른 진행도 계산
        """
        if self.is_in_warmup(epoch):
            return 0.0

        adjusted_epoch = epoch - self.warmup_epochs
        if self.penalty_warmup > 0:
            return min(1.0, adjusted_epoch / self.penalty_warmup)
        return 1.0

    def get_logits(self, image):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        return logits_student, logits_teacher

    def forward_train(self, image, target, epoch=None, **kwargs):
        logits_student, logits_teacher = self.get_logits(image)

        # 1. Compute standard Cross-Entropy Loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        # 2. Compute KD loss per sample
        per_sample_kd_loss = self.kd_loss_weight * kd_loss(
            logits_student, logits_teacher, self.temperature, self.logit_stand, reduction="none"
        )

        # 3. Apply penalty reweighting if:
        # - Using "kdas" mode
        # - Not in warmup period
        # - Penalty parameters are set
        if (self.cfg.SOLVER.TRAINER == "kdas" and
                not self.is_in_warmup(epoch) and
                self.penalty_factor is not None):

            # Compute teacher-target divergence
            num_classes = logits_teacher.size(1)
            eps = 1e-6
            smoothed_targets = torch.full_like(logits_teacher, eps / (num_classes - 1))
            smoothed_targets.scatter_(1, target.unsqueeze(1), 1 - eps)
            teacher_log_prob = F.log_softmax(logits_teacher / 1, dim=1)
            per_sample_kl_tt = F.kl_div(
                teacher_log_prob, smoothed_targets, reduction="none"
            ).sum(dim=1) * (1 ** 2)

            # Thresholds: 20th and 80th percentiles
            lower_bound = torch.quantile(per_sample_kl_tt, self.cfg.KDAS.THRESHOLD)
            upper_bound = torch.quantile(per_sample_kl_tt, 1 - self.cfg.KDAS.THRESHOLD)

            # Get penalty progress (considering both warmup periods)
            dynamic_factor = self.get_penalty_progress(epoch)
            effective_lambda = dynamic_factor * self.penalty_lambda

            # Compute weights
            weight = torch.ones_like(per_sample_kl_tt)
            mask_low = per_sample_kl_tt < lower_bound
            mask_high = per_sample_kl_tt > upper_bound
            weight[mask_low] = 1 - effective_lambda * (lower_bound - per_sample_kl_tt[mask_low])
            weight[mask_high] = 1 - effective_lambda * (per_sample_kl_tt[mask_high] - upper_bound)
            weight = torch.clamp(weight, min=self.penalty_factor, max=1.0)
            loss_kd = (per_sample_kd_loss * weight).mean()
        else:
            # During warmup: use standard KD loss
            loss_kd = per_sample_kd_loss.mean()

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd
        }
        return logits_student, losses_dict
