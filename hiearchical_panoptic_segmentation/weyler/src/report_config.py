""" Configuration file.
"""
import copy
import os

DATASET_DIR = os.environ.get('DATASET_DIR', '</path/to/dataset>')
LOG_DIR = os.environ.get('WEYLER_LOG_DIR', '<path/to/log/directoy>')

def env_int(name, default):
  value = os.environ.get(name)
  return default if value is None else int(value)

args = dict(
  train_img_dir=os.path.join(DATASET_DIR, 'train', 'images'),
  
  val_img_dir=os.path.join(DATASET_DIR, 'val', 'images'),

  test_img_dir=os.path.join(DATASET_DIR, 'test', 'images'),

  report_dir = os.path.join(LOG_DIR, 'reports'),

  type=os.environ.get('WEYLER_SPLIT', 'test'),
  
  width = env_int('WEYLER_WIDTH', 1024),
  height = env_int('WEYLER_HEIGHT', 1024),

  n_classes = 1,
  n_sigma = 3,
  apply_offsets = True,

  sigma_scale = 11.0,
  alpha_scale = 11.0,

  parts_area_thres=32,
  parts_score_thres=0.7,

  objects_area_thres = 64,
  objects_score_thres = 0.7,

  cls_colors = {"0" : "#ff0000", "1" : "#1eff00"}
)

def get_args():
  return copy.deepcopy(args)
