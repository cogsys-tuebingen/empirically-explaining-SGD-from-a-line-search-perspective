import torch.optim as opt
from data_sampling.source import PalOptimizer
from data_sampling.source.optimizers.sgd_line_sampler import SGDLineSampler
from data_sampling.source.optimizers.sgd_line_sampler_on_position import SGDLineSamplerOnPosition

optimizer_dict = {
    "SGD": opt.SGD,
    "SGD_LS": SGDLineSampler,
    "SGD_LS_OP": SGDLineSamplerOnPosition,
    "PAL": PalOptimizer,
    "ADAM": opt.Adam,
    "RMSPROP": opt.RMSprop,
    "ADAGRAD": opt.Adagrad,
    "ADADELTA": opt.Adadelta,
}
