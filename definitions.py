"""
Module to contain the path of the ROOT DIR
"""

import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
WORK_DIR = Path(ROOT_DIR / 'work_files')
WORK_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(ROOT_DIR / 'data')
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(ROOT_DIR / 'results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

project_folders = {'root': ROOT_DIR,
                   'work': WORK_DIR,
                   'result': RESULTS_DIR}
