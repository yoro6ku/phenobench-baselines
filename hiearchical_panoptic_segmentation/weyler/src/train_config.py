"""
Set training options.
"""
import copy
import os

import torch
from utils import mytransforms as my_transforms

DATASET_DIR=os.environ.get('DATASET_DIR')

def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)

args = dict(

    cuda=env_bool('WEYLER_CUDA', True),

    save=True,
    save_dir=os.environ.get('WEYLER_SAVE_DIR', '<path/to/save/directoy>'),
    resume_path=os.environ.get('WEYLER_RESUME_PATH', '<path/to/checkpoint.pth>'),

    only_eval=env_bool('WEYLER_ONLY_EVAL', True), # set to False if you want to train a new model

    log_dir=os.environ.get('WEYLER_LOG_DIR', '<path/to/log/directoy>'),

    train_dataset = {
        'name': 'mydataset',
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': os.environ.get('WEYLER_TRAIN_SPLIT', 'train'),
            'size': None,
            'stems': False,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'global_instances', 'global_labels', 'parts_instances', 'parts_labels'],
                        'type': [torch.FloatTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': env_int('WEYLER_WORKERS', 8)
    },

    val_dataset = {
        'name': 'mydataset',
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': os.environ.get('WEYLER_SPLIT', 'test'), # 'val' or 'test'
            'stems': False,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image'],
                        'type': [torch.FloatTensor]
                    }
                },
            ]),
        },
        'batch_size': env_int('WEYLER_BATCH_SIZE', 1),
        'workers': env_int('WEYLER_WORKERS', 8)
    },

    image = {
        'im_width': env_int('WEYLER_WIDTH', 1024),
        'im_height': env_int('WEYLER_HEIGHT', 1024),
    },

    model = {
        'name': 'branched_erfnet',
        'kwargs': {
            'num_classes': [2*5,1*2],
            'batch_norm': True,
            'instance_norm': False,
        }
    }, 

    lr= 1e-3,
    w_decay=0,
    n_epochs=512,
    report_epoch=127, # every x epochs, report train & validation set

    # fix model params
    sigma_scale = 11.0,
    alpha_scale = 11.0,

    # loss options
    loss_opts={
        'to_center': True,
        'apply_offsets': True,
        'n_sigma': 3,
        'class_weights': [10],
        'label_ids': [1]
    },

    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
        'w_offset': 0,
    },
)

def get_args():
  return copy.deepcopy(args)
