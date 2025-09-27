#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script extracts the data for Fig2 B/C and Fig3C/D and stores it in svg files
Additionally, the plots for Fig2D, Fig3A/B/F/G/ are created. Everything is stored in "Results/Fig2_mscoco/"
"""

# imports 

import scipy.stats
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from functions import get_device
import scipy
import ClosedFormDecoding

parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()

DEVICE = get_device()

MNIST = False
USE_RES_NET = False

R_PATH = 'EmergentPredictiveCoding/Results/Fig2/Data/'
if MNIST:
    F_PATH = 'Results/Fig2/'
else:
    F_PATH = 'EmergentPredictiveCoding/Results/Fig2_mscoco/'
if MNIST:
    M_PATH = 'patterns_rev/seeded_mnist/'
else:
    M_PATH = 'patterns_rev/mscoco_deepgaze3/'

hdf_path = R_PATH+'network_stats.h5'

LOAD = False
SEED = 2553
if not os.path.isdir(os.path.dirname(R_PATH)):
    os.makedirs(os.path.dirname(R_PATH), exist_ok=True)
if not os.path.isdir(os.path.dirname(F_PATH)):
    os.makedirs(os.path.dirname(R_PATH), exist_ok=True)
    
if SEED != None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
# set up hdf5 file to store the results 
if not os.path.exists(hdf_path):
    store = pd.HDFStore(hdf_path)
    store.close()
INPUT_SIZE = 128*128
HIDDEN_SIZE = 2048
if MNIST:
    INPUT_SIZE = 54*54
    HIDDEN_SIZE = 54*54
Z_CRIT = 2.576 #99%
SEQ_LENGTH = 10
TIME_STEPS_IMG = 6
TIME_STEPS_CORDS = 3
# dataset loaders
import mnist
from H5dataset import H5dataset

# framework files
import RNN
import plot
from matplotlib.ticker import MaxNLocator


# dataset loaders
if MNIST:
    train_set, validation_set, test_set = mnist.load(val_ratio=0.0)
    validation_set = test_set
else:
    h5_dataset = '/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5'
    validation_set = H5dataset('val', h5_dataset, device=DEVICE, use_color=USE_RES_NET, class_labels=False)
    test_set = H5dataset('test', h5_dataset, device=DEVICE, use_color=USE_RES_NET, class_labels=False)
    train_set = H5dataset('train', h5_dataset, device=DEVICE, use_color=USE_RES_NET, class_labels=False)

# load pre, post MNIST networks
nets = [[], [], [], [], []]

NUM_INSTANCES = 1
USE_CONV = False 
WARP_IMGS = False
USE_LSTM = False
LESION_PRED_UNITS = True
LESION_RANDOM = True
# Load multiple models with the same loss function (e.g. for lesioning)
losses = ['l1_all', 'l1_all', 'l1_all']

if USE_CONV and len(losses) == 1:
    losses = [loss + 'conv' for loss in losses]
# set up dictionaries to fill in the data
ec_results, ap_results, st_results, pre_results = dict(), dict(), dict(), dict()
result_list = [('ec', ec_results),('ap', ap_results), ('st', st_results), ('pre', pre_results)]
if MNIST:
    net_name = "mnist_net"
else:
    net_name = "mscoco_net"

# ==============================================
# Initialize all models
# ==============================================

loss_ind = 0
loss = losses[0]
i = 0

single_layer_net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=4096,
                title=M_PATH+net_name+loss,
                device=DEVICE,
                use_fixation=(i==0),
                use_conv=('conv' in loss),
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=TIME_STEPS_IMG,
                time_steps_cords=TIME_STEPS_CORDS,
                mnist=MNIST,
                twolayer=(loss_ind!=0),
                dropout=0,
                disentangled_loss=False,
                useReservoir=False,
                num_layers=1,
                supervised=False)
single_layer_net.load(i, twolayers=True)

three_layer_net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=1024,
                title=M_PATH+net_name+loss,
                device=DEVICE,
                use_fixation=(i==0),
                use_conv=('conv' in loss),
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=TIME_STEPS_IMG,
                time_steps_cords=TIME_STEPS_CORDS,
                mnist=MNIST,
                twolayer=(loss_ind!=0),
                dropout=0,
                disentangled_loss=False,
                useReservoir=False,
                num_layers=3,
                supervised=False)
three_layer_net.load(i, twolayers=True)

short_net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                title=M_PATH+net_name+loss,
                device=DEVICE,
                use_fixation=(i==0),
                use_conv=('conv' in loss),
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=4,
                time_steps_cords=2,
                mnist=MNIST,
                twolayer=(loss_ind!=0),
                dropout=0,
                disentangled_loss=False,
                useReservoir=False,
                num_layers=2,
                supervised=False)
short_net.load(i, twolayers=True)

long_net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                title=M_PATH+net_name+loss,
                device=DEVICE,
                use_fixation=(i==0),
                use_conv=('conv' in loss),
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=8,
                time_steps_cords=4,
                mnist=MNIST,
                twolayer=(loss_ind!=0),
                dropout=0,
                disentangled_loss=False,
                useReservoir=False,
                num_layers=2,
                supervised=False)
long_net.load(i, twolayers=True)

supervised_net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                title=M_PATH+net_name+loss,
                device=DEVICE,
                use_fixation=(i==0),
                use_conv=('conv' in loss),
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=TIME_STEPS_IMG,
                time_steps_cords=TIME_STEPS_CORDS,
                mnist=MNIST,
                twolayer=(loss_ind!=0),
                dropout=0,
                disentangled_loss=False,
                useReservoir=False,
                num_layers=2,
                supervised=True)
supervised_net.load(i, twolayers=True)

def compute_mean_ci(data):
    data = data.flatten()
    mean, se = np.mean(data), scipy.stats.sem(data)
    h = se * scipy.stats.t.ppf((1 + 0.99) / 2., data.shape[0]-1)
    return mean, mean-h, mean+h


# ==============================================
# extract test losses of different models
# ==============================================
print("test losses:")
print("supervised")
torch.manual_seed(SEED)
losses_supervised, _, _ = plot.compare_previous_fixation(supervised_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, returnFeedback=True)
print("Averaged fixation onset: ", compute_mean_ci(losses_supervised[:, 1:, 0]))
print("8 timesteps")
torch.manual_seed(SEED)
losses_long, _, _ = plot.compare_previous_fixation(long_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, returnFeedback=True)
print("Averaged fixation onset: ", compute_mean_ci(losses_long[:, 1:, 0]))
print("4 timesteps")
torch.manual_seed(SEED)
losses_short, _, _ = plot.compare_previous_fixation(short_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, returnFeedback=True)
print("Averaged fixation onset: ", compute_mean_ci(losses_short[:, 1:, 0]))
print("1 hidden layer")
torch.manual_seed(SEED)
losses_single_layer, _, _ = plot.compare_previous_fixation(single_layer_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, returnFeedback=True)
print("Averaged fixation onset: ", compute_mean_ci(losses_single_layer[:, 1:, 0]))
print("3 hidden layer")
torch.manual_seed(SEED)
losses_three_layer, _, _ = plot.compare_previous_fixation(three_layer_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, returnFeedback=True)
print("Averaged fixation onset: ", compute_mean_ci(losses_three_layer[:, 1:, 0]))


# ==============================================
# Extract decoding performance of different models
# ==============================================
print("decoding:")
if LESION_PRED_UNITS:
    print("supervised")
    torch.manual_seed(SEED)
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(supervised_net, validation_set, validation_set, mode='global', timestep=None)
    print("8 timesteps")
    torch.manual_seed(SEED)
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(long_net, validation_set, validation_set, mode='global', timestep=None)
    print("4 timesteps")
    torch.manual_seed(SEED)
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(short_net, validation_set, validation_set, mode='global', timestep=None)
    print("1 hidden layer")
    torch.manual_seed(SEED)
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(single_layer_net, validation_set, validation_set, mode='global', timestep=None, layer=[1])
    print("3 hidden layer")
    torch.manual_seed(SEED)
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(three_layer_net, validation_set, validation_set, mode='global', timestep=None, layer = [1, 2, 3])
