"""
 Created by Myung-Joon Kwon
 mjkwon2021@gmail.com
 July 7, 2020
"""
from pathlib import Path

project_root = Path(__file__).parent
dataset_root = Path("/kaggle/input")
dataset_paths = {
    'COCO':dataset_root / "psccnet-data", 
    'SAVE_PRED': project_root / "output_pred"
}



