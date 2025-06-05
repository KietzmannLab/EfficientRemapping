import argparse
import mnist
import RNN
import random
import torch
from functions import get_device
from train import train
from H5dataset import H5dataset
from DecodingModel import DecodingModelState
from train import test_epoch
import plot
parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()
scaler = None
MNIST = False


DEVICE = get_device()

INPUT_SIZE = 128*128
HIDDEN_SIZE = 2048
if MNIST:
    INPUT_SIZE = 54*54
    HIDDEN_SIZE = 54*54
BATCH_SIZE = 1024
NUM_EPOCHS = 25
SEQ_LENGTH = 10
USE_CONV = False
WARP_IMGS = False
USE_RES_NET = False
TIME_STEPS_IMG = 6
TIME_STEPS_CORDS = 3

TIMESTEP = 0
USE_PRED = False

# dataset loaders
if MNIST:
    train_set, validation_set, test_set = mnist.load(val_ratio=0.0)
else:
    h5_dataset = '/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5'
    validation_set = H5dataset('test', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    test_set = H5dataset('val', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    train_set = H5dataset('train', h5_dataset, device=DEVICE, use_color=USE_RES_NET)


"""
Load and test decoding model
"""
figax = None
figax_rsquared = None

loader = test_set.create_batches(batch_size=1024, shuffle=True)
num_batches = len(test_set) // 1024
mean_fix = torch.zeros((1,2))
for batch, fixation in loader:
    with torch.no_grad():
        mean_fix += torch.mean(fixation, dim=[0, 1])
mean_fix /= num_batches
# print(f"Mean fixation: {mean_fix}")


for TIMESTEP in range(6):
    n_instances = 1 # number of model instances
    # losses = ['l1_pre']
    losses = ['l1_all']
    if USE_CONV:
        losses = [loss + 'conv' for loss in losses]
    seeds = [[random.randint(0,10000) for i in range(n_instances)] for j in range(len(losses))]

    for loss_ind, loss in enumerate(losses):
        for i in range(0, n_instances):
            net = RNN.State(activation_func=torch.nn.ReLU(),
                    optimizer=torch.optim.Adam,
                    # lr=8e-5,
                    lr=1e-4,
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    # title="patterns_rev/mscoco_deepgaze3/mscoco_net"+loss+"_"+str(i)+"_excitatory_large_kernels",
                    title='patterns_rev/mscoco_deepgaze3/mscoco_net' + loss,
                    # title=f"patterns_rev/seeded_mnist/mnist_net{loss}_{i}_fc_1layer_shuffled_positions_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_ordered_1init_largerLR_",
                    device=DEVICE,
                    use_fixation=(i == 0),
                    seed=seeds[loss_ind][i],
                    use_conv=USE_CONV,
                    warp_imgs=WARP_IMGS,
                    use_resNet=USE_RES_NET,
                    time_steps_img=TIME_STEPS_IMG,
                    time_steps_cords=TIME_STEPS_CORDS,
                    mnist=MNIST)
            net.load(i)
            losses_current = []
            losses_next = []
            rsquared_current = []
            rsquared_next = []
            for timestep in range(6):
            # for timestep in [None]:
                for test_next in [False, True]:
                    for layer in [2]:
                    # for layer in [1, 2]:
                        decoding_model = DecodingModelState(torch.optim.Adam,
                                                    1e-4,
                                                    # f"patterns_rev/mscoco_deepgaze3/decoding_model__timestep_{TIMESTEP}layer{layer}_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_clipping_1000_current_",
                                                    # f"patterns_rev/mscoco_deepgaze3/decoding_model__timestep_{TIMESTEP}predUnits_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_clipping_1500_current_",
                                                    # f"patterns_rev/mscoco_deepgaze3/decoding_model_predUnits_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_clipping_1500_current_",
                                                    # f"patterns_rev/mscoco_deepgaze3/decoding_model_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_layer{layer}_1500_current_",
                                                    # f"patterns_rev/mscoco_deepgaze3/decoding_model_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_layer{layer}_1500_current_pred{USE_PRED}_",
                                                    f"patterns_rev/mscoco_deepgaze3/decoding_model_timestep_{TIMESTEP}_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_layer{layer}_1500_current_pred{USE_PRED}_",
                                                    DEVICE,
                                                    net,
                                                    layer,
                                                    HIDDEN_SIZE,
                                                    time_step=timestep,
                                                    test_next_fixation=test_next,
                                                    use_pred_units=USE_PRED,
                                                    test_set=test_set)
                        decoding_model.load()
                        decoding_model.set_mean_fix(mean_fix=mean_fix)
                        # untrained_decoding_model = DecodingModelState(torch.optim.Adam,
                        #                             1e-4,
                        #                             f"",
                        #                             DEVICE,
                        #                             net,
                        #                             layer,
                        #                             HIDDEN_SIZE,
                        #                             time_step=timestep,
                        #                             test_next_fixation=test_next)
                        if test_next:
                            print(f'Test loss decoding model layer {layer}, next fixation, timestep {timestep}')
                            next_loss, next_r_squared = test_epoch(decoding_model, test_set, None, 1024, 10, mnist=False)
                            losses_next.append(next_loss)
                            rsquared_next.append(next_r_squared)
                            print("Test r_squared:     {:.8f}".format(next_r_squared))
                            # print(f'Test loss untrained decoding model layer {layer}, next fixation, timestep {timestep}')
                            # test_epoch(untrained_decoding_model, test_set, None, 1024, 10, mnist=False)
                        else:
                            print(f'Test loss decoding model layer {layer}, current fixation, timestep {timestep}')
                            current_loss, current_r_squared = test_epoch(decoding_model, test_set, None, 1024, 10, mnist=False)
                            losses_current.append(current_loss)
                            rsquared_current.append(current_r_squared)
                            print("Test r_squared:     {:.8f}".format(current_r_squared))
                            # print(f'Test loss untrained decoding model layer {layer}, current fixation, timestep {timestep}')
                            # test_epoch(untrained_decoding_model, test_set, None, 1024, 10, mnist=False)
            figax = plot.linePlot([range(6), range(6)], [losses_current, losses_next], color=TIMESTEP, linestyle=['-', ':'], label=[f"Time step {TIMESTEP}, current", f"Time step {TIMESTEP}, next"], figax=figax)
            figax_rsquared = plot.linePlot([range(6), range(6)], [rsquared_current, rsquared_next], color=TIMESTEP, linestyle=['-', ':'], label=[f"Time step {TIMESTEP}, current", f"Time step {TIMESTEP}, next"], figax=figax_rsquared, ylim=(0, 1))
            # figax = plot.linePlot([range(6), range(6)], [losses_current, losses_next], color=0, linestyle=['-', ':'], label=[f"Current", f"Next"], figax=figax)
            # figax_rsquared = plot.linePlot([range(6), range(6)], [rsquared_current, rsquared_next], color=0, linestyle=['-', ':'], label=[f"Current", f"Next"], figax=figax_rsquared, ylim=(0, 1))
            # figax_rsquared = plot.scatter([range(6), range(6)], [[max(0, elem) for elem in rsquared_current], [max(0, elem) for elem in rsquared_next]], color=TIMESTEP, figax=figax_rsquared)
plot.save_fig(figax[0], 'EmergentPredictiveCoding/Results/Fig2_mscoco/lossesDecoders', bbox_inches='tight')
plot.save_fig(figax_rsquared[0], 'EmergentPredictiveCoding/Results/Fig2_mscoco/rSquaredDecoders', bbox_inches='tight')
