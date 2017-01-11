#!/usr/bin/env python3

import os

path = os.path.abspath(__file__)
src_folder = os.path.dirname(path)
MAIN_FOLDER = os.path.dirname(src_folder)
MODELS_FOLDER = os.path.join(MAIN_FOLDER, 'models/')
DATA_FOLDER = os.path.join(MAIN_FOLDER, 'data/')
