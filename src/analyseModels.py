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
    validation_set = H5dataset('val', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    test_set = H5dataset('test', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    train_set = H5dataset('train', h5_dataset, device=DEVICE, use_color=USE_RES_NET)

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
for loss_ind, loss in enumerate(losses):
    for i in range(0, NUM_INSTANCES):
        net = RNN.State(activation_func=torch.nn.ReLU(),
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
                dropout=0)
        net.load(i, twolayers=loss_ind!=0)
        nets[loss_ind].append(net)

small_fixation_net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=85*85,
                hidden_size=HIDDEN_SIZE,
                title=M_PATH+net_name+loss,
                device=DEVICE,
                use_fixation=True,
                use_conv=False,
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=TIME_STEPS_IMG,
                time_steps_cords=TIME_STEPS_CORDS,
                mnist=MNIST,
                twolayer=True,
                dropout=0)
small_fixation_net.load(i, twolayers=True)
nets[3].append(small_fixation_net)

no_efference_net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                title=M_PATH+net_name+loss,
                device=DEVICE,
                use_fixation=False,
                use_conv=False,
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=TIME_STEPS_IMG,
                time_steps_cords=TIME_STEPS_CORDS,
                mnist=MNIST,
                twolayer=True,
                dropout=0)
no_efference_net.load(i, twolayers=True)
nets[4].append(no_efference_net)

plot.decoding_results()
plot.decoding_example()
loss_idx = 0
trained_net = nets[loss_idx][0] # example trained network for visualisation

untrained_net = RNN.State(activation_func=torch.nn.ReLU(),
        optimizer=torch.optim.Adam,
        lr=1e-4,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        title="",
        device=DEVICE,
        use_conv=USE_CONV,
        use_lstm=USE_LSTM,
        warp_imgs=WARP_IMGS,
        use_resNet=USE_RES_NET,
        time_steps_img=TIME_STEPS_IMG,
        time_steps_cords=TIME_STEPS_CORDS,
        mnist=MNIST)


# ==============================================
# extract correlation matrix Fig 3F not lesioned
# ==============================================
torch.manual_seed(SEED)
_, _, feedback_grouped = plot.compare_previous_fixation(trained_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, returnFeedback=True)
similarity_full, corrs_full = plot.pred_rdm(feedback_grouped[:, 0], feedback_grouped[:, 1])


# ==============================================
# Train decoding model on trained and untrained model, lesion trained net
# ==============================================
if LESION_PRED_UNITS:
    torch.manual_seed(SEED)
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(untrained_net, train_set, validation_set, mode='global', timestep=None)
    torch.manual_seed(SEED)
    pred_cells, reg_weights, test_score = ClosedFormDecoding.regressionCoordinates(trained_net, train_set, validation_set, mode='global', timestep=None)
    trained_net.model.setLesionMap(pred_cells, random=False)

    df = pd.DataFrame(reg_weights.T, columns=['x', 'y'])
    df.to_csv(F_PATH+"decoder_weights.csv")

plot.checkCellAvgActivity(trained_net, validation_set)

# ==============================================
# Extract loss/ energy consumption for all controls (Fig2 A, B) and for lesioned model (Fig3D)
# ==============================================

# compare real fixation order to shuffled fixation order as coordinate input to the model
torch.manual_seed(SEED)
losses, losses_random_fix, fixations = plot.compare_random_fixations(trained_net, validation_set, loss_fn=losses[loss_idx], use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, mnist=MNIST, return_fixations=True)
# torch.manual_seed(SEED)
# _, losses_random_fix_just_onset = plot.compare_random_fixations(trained_net, validation_set, loss_fn=losses[loss_idx], use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, mnist=MNIST, just_onset=True)

torch.manual_seed(SEED)
_, losses_avg_img, feedback = plot.compare_average_img(trained_net, validation_set, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, return_feedback=True)
plot.checkLossesInhibitory(feedback)

torch.manual_seed(SEED)
_, losses_avg_local_crop, _ = plot.compare_avg_local_crop(trained_net, validation_set, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, return_feedback=True)

torch.manual_seed(SEED)
_, _, feedback_grouped = plot.compare_previous_fixation(trained_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, returnFeedback=True)
similarity_lesioned, corrs_lesioned = plot.pred_rdm(feedback_grouped[:, 0], feedback_grouped[:, 1])

plot.makeRDMs(similarity_full, similarity_lesioned, corrs_full, corrs_lesioned)
torch.manual_seed(SEED)
losses_avg_color = plot.compare_average_color(trained_net, validation_set, train_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET)

torch.manual_seed(SEED)
losses_without_first, losses_prev_fix = plot.compare_previous_fixation(trained_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET)

torch.manual_seed(SEED)
losses_fix_onset, losses_prev_fix_onset = plot.compare_previous_fixation(trained_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, just_fixation_onset=True)

torch.manual_seed(SEED)
plot.extract_energy_usage(trained_net, validation_set)

torch.manual_seed(SEED)
plot.extract_loss_timeseries(trained_net, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, include_initial=True, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE)

# ==============================================
# Save fixations for later analysis (FIG 2C)
# ==============================================
fixations =fixations.reshape(fixations.shape[0], 2*7)
df = pd.DataFrame(fixations, columns=np.concatenate((np.expand_dims(['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', ], axis=0), np.expand_dims([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], 0)), axis=0).T)
df.to_csv(F_PATH+"fixations.csv")


# ==============================================
# Extract loss/ energy consumption for other models (Fig 2B/ 3D)
# ==============================================

# random lesion
if len(nets[1]) != 0:
    trained_net_all = nets[1][0]
    if LESION_RANDOM:
        trained_net_all.model.setLesionMap(random=True)
    USE_CONV = trained_net_all.model.use_conv
    torch.manual_seed(SEED)
    losses_all, losses_random_fixations_all = plot.compare_random_fixations(trained_net_all, validation_set, loss_fn=losses[loss_idx], use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, mnist=MNIST)

# no lesion
if len(nets[2]) != 0:
    trained_net_noLesion = nets[2][0]
    USE_CONV = trained_net_noLesion.model.use_conv
    torch.manual_seed(SEED)
    losses_no_lesion, losses_random_fixations_no_lesion = plot.compare_random_fixations(trained_net_noLesion, validation_set, loss_fn=losses[loss_idx], use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, mnist=MNIST)
    _, _, feedback = plot.compare_average_img(trained_net_noLesion, validation_set, validation_set, use_conv=USE_CONV, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, return_feedback=True)
    plot.checkLossesInhibitory(feedback)

# small crops
if len(nets[3]) != 0:
    trained_net_small_crops = nets[3][0]
    torch.manual_seed(SEED)
    losses_small_crops, _ = plot.compare_random_fixations(trained_net_small_crops, validation_set, loss_fn=losses[loss_idx], use_conv=False, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, mnist=MNIST)

# no efference copy
if len(nets[4]) != 0:
    trained_net_no_efference = nets[4][0]
    torch.manual_seed(SEED)
    losses_no_efference, _ = plot.compare_random_fixations(trained_net_no_efference, validation_set, loss_fn=losses[loss_idx], use_conv=False, warp_imgs=WARP_IMGS, use_resNet=USE_RES_NET, feature_size=INPUT_SIZE, mnist=MNIST)


# ==============================================
# save all losses for plotting in csv
# ==============================================
columns = [np.repeat(range(7), 6),
           list(range(6)) + list(range(6)) + list(range(6)) + list(range(6)) + list(range(6)) + list(range(6)) + list(range(6))]
columns = np.concatenate([np.expand_dims(np.repeat(["Model", "Model lesion", "Model random Lesion", "Model small crops", "Model no efference", "Avg image", "Avg local crop", "Avg color", "Prev Fixation", "Shuffled Fixation", "Shuffled fixation Lesioned", "Shuffled fixation random Lesion"], 42), 0), np.concatenate([columns, columns, columns, columns, columns, columns, columns, columns, columns, columns, columns, columns], axis=1)], axis=0)
columns = columns.transpose(1, 0)
# print(columns.shape)
data = np.stack([losses_no_lesion, losses, losses_all, losses_small_crops, losses_no_efference, losses_avg_img, losses_avg_local_crop, losses_avg_color, losses_prev_fix, losses_random_fixations_no_lesion, losses_random_fix, losses_random_fixations_all], axis=1)
data = data.reshape(data.shape[0], 12*6*7)
# print(data.shape)
df = pd.DataFrame(data, columns=columns)
# print(df.shape)
df.to_csv(F_PATH+"losses.csv")



# ==============================================
# get feedback visualisations for trained lesioned network
# ==============================================
X, P, _, T, C = plot.example_sequence_state(trained_net, validation_set, mnist=MNIST, use_conv=USE_CONV, warp_imgs=WARP_IMGS)


fixation_onsets = [P[i*6] for i in range(7)]
ground_truths = [X[i*6] * -1 for i in range(7)]

examples = [ground_truths[i//2]if i % 2 == 0 else fixation_onsets[i//2 ] for i in range(14)]

fig, axes = plot.display(ground_truths + fixation_onsets, lims=None, shape=(7,2), figsize=(8, 2), axes_visible=False, layout='tight')
plot.save_fig(fig, F_PATH+"example_predictions_lesion", bbox_inches='tight')


# ==============================================
# get feedback visualisations for trained full network
# ==============================================
X, P, _, T, C = plot.example_sequence_state(trained_net_noLesion, validation_set, mnist=MNIST, use_conv=USE_CONV, warp_imgs=WARP_IMGS)


fixation_onsets = [P[i*6] for i in range(7)]
ground_truths = [X[i*6] * -1 for i in range(7)]

examples = [ground_truths[i//2]if i % 2 == 0 else fixation_onsets[i//2 ] for i in range(14)]

fig, axes = plot.display(ground_truths + fixation_onsets, lims=None, shape=(7,2), figsize=(8, 2), axes_visible=False, layout='tight')
plot.save_fig(fig, F_PATH+"example_predictions_no_lesion", bbox_inches='tight')

