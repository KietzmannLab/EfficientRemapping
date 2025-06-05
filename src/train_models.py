import argparse
import mnist
import RNN
import random
import torch
from functions import get_device
from train import train
from H5dataset import H5dataset
import os.path
import wandb

wandb.login()

parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()
# scaler =torch.cuda.amp.GradScaler()
scaler = None
MNIST = False


DEVICE = get_device()

INPUT_SIZE = 128*128
# INPUT_SIZE = 85*85
HIDDEN_SIZE = 2048
if MNIST:
    INPUT_SIZE = 54*54
    HIDDEN_SIZE = 54*54
BATCH_SIZE = 1024
NUM_EPOCHS = 1500
SEQ_LENGTH = 10
USE_CONV = False
USE_LSTM = False
WARP_IMGS = False
USE_RES_NET = False
TIME_STEPS_IMG = 6
TIME_STEPS_CORDS = 3
# LEARNING_RATE = 5e-4
DISENTANGLED_LOSS = False
LEARNING_RATE = 7e-4
DROPOUT = 0
USE_RESERVOIR = True


# dataset loaders
if MNIST:
    train_set, validation_set, test_set = mnist.load(val_ratio=0.0)
else:
    h5_dataset = '/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5'
    validation_set = H5dataset('test', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    test_set = H5dataset('val', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    train_set = H5dataset('train', h5_dataset, device=DEVICE, use_color=USE_RES_NET)


"""
Create and train ten instances of energy efficient RNNs for MNIST 
"""
n_instances = 1 # number of model instances
# losses = ['l1_pre']
losses = ['l1_all']
# losses = ['l1_pre', 'l1_all']
if USE_CONV:
    losses = [loss + 'conv' for loss in losses]
seeds = [[random.randint(0,10000) for i in range(n_instances)] for j in range(len(losses))]
#seeds = [[random.randint(0,10000) for i in range(n_instances)]]
# train MNIST networks

for loss_ind, loss in enumerate(losses):
    run = wandb.init(
        project='efficient_remapping',
        config={'learning_rate':LEARNING_RATE,
                'epochs': NUM_EPOCHS,
                'scheduler decay': 0.75,
                'changed_LR': True,
                'dataset': 'MSCoco',
                'LSTM': USE_LSTM,
                'loss': loss,
                'hidden_size': HIDDEN_SIZE,
                'dropout': DROPOUT,
                'gradient_clipping': True,
                'disentangled_loss': DISENTANGLED_LOSS}
    )
    for i in range(0, n_instances):
#     for i in [1]:
        print("loss", loss_ind, "instance", i)
        net = RNN.State(activation_func=torch.nn.ReLU(),
                optimizer=torch.optim.Adam,
                lr=LEARNING_RATE,
                # lr=1e-4,
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                # title="patterns_rev/mscoco_deepgaze3/mscoco_net"+loss+"_"+str(i)+"_excitatory_large_kernels",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_lstm_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_1e4_scheduler_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_conv_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_clipping_multistep_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_conv_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_ReLU_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_hybrid_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr5e5_partial_ReLU_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_hybrid_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_clipping_multistep_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr3e4_clipping_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr2e4_global_ReLU_disentangled_loss_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_ReLU_nonCords_new_moreRL_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_ReLU_nonCords_new_moreRL_noCords_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_ReLU_nonCords_new_moreRL_smallCrops_",
                title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_ReLU_nonCords_new_moreRL_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr5e4_ReLU_moreDecay_",
                # title=f"patterns_rev/mscoco_deepgaze3/mscoco_net{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_lr1e4_50_6_grid_cells_",
                # title=f"patterns_rev/seeded_mnist/mnist_net{loss}_{i}_fc_1layer_shuffled_positions_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_ordered_1init_largerLR_",
                device=DEVICE,
                use_fixation=True,
                # seed=seeds[loss_ind][i],
                use_conv=USE_CONV,
                use_lstm=USE_LSTM,
                warp_imgs=WARP_IMGS,
                use_resNet=USE_RES_NET,
                time_steps_img=TIME_STEPS_IMG,
                time_steps_cords=TIME_STEPS_CORDS,
                mnist=MNIST,
                dropout=DROPOUT,
                disentangled_loss=DISENTANGLED_LOSS,
                useReservoir=USE_RESERVOIR)
        
        start_epoch = 0
        for epochs in range(0, NUM_EPOCHS, 50):
            path = f"./EmergentPredictiveCoding/models/{net.title}{epochs}.pth"
            print(path, os.path.exists(path))
            if os.path.exists(path):
                net.load(path=path)
                start_epoch = epochs
      
        train(net,
              train_ds=train_set,
              test_ds=test_set,
              loss_fn=loss,
              num_epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE,
              sequence_length=SEQ_LENGTH,
              verbose=False,
              mnist=MNIST,
              scaler=scaler,
              start_epoch=start_epoch)
            
        # save model
        net.save(NUM_EPOCHS)
    # wandb.finish()
