from .trainer import BaseTrainer, CRDTrainer, AugTrainer, KDASTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "augs": AugTrainer,
    "kdas": KDASTrainer
}
