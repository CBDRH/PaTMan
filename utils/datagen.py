import os
import tensorflow as tf

MODEL_INPUT_DATAPATH = os.path.join(os.path.join(os.getcwd(), 'inputfiles'))
if not os.path.exists(MODEL_INPUT_DATAPATH):
    os.makedirs(MODEL_INPUT_DATAPATH)
    
PROCESSED_DATAPATH = os.path.join(os.path.join(os.getcwd(), 'processeddata'))

if not os.path.exists(PROCESSED_DATAPATH):
    os.makedirs(PROCESSED_DATAPATH)
    
RESULTFILE_DATAPATH = os.path.join(os.path.join(os.getcwd(), 'resultfiles'))

if not os.path.exists(RESULTFILE_DATAPATH):
    os.makedirs(RESULTFILE_DATAPATH)
    