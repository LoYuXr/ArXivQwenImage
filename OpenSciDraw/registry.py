
from mmengine.registry import Registry, build_model_from_cfg

MODELS = Registry('model', build_model_from_cfg, locations=['OpenSciDraw.models'])
DATASETS = Registry('dataset', build_model_from_cfg, locations=['OpenSciDraw.datasets'])
TRAIN_ITERATION_FUNCS = Registry('train_iteration_funcs', locations=['OpenSciDraw.train_iteration_funcs'])
VALIDATION_FUNCS = Registry('validation_funcs', locations=['OpenSciDraw.validation_funcs'])