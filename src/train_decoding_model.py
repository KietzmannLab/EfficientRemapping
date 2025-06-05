import argparse
import mnist
import RNN
import random
import torch
from functions import get_device
from train import train
from H5dataset import H5dataset
from DecodingModel import DecodingModelState

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

TIMESTEP = 4
NEXT_FIXATION = False
USE_PRED_UNITS = False

# dataset loaders
if MNIST:
    train_set, validation_set, test_set = mnist.load(val_ratio=0.0)
else:
    h5_dataset = '/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5'
    validation_set = H5dataset('test', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    test_set = H5dataset('val', h5_dataset, device=DEVICE, use_color=USE_RES_NET)
    train_set = H5dataset('train', h5_dataset, device=DEVICE, use_color=USE_RES_NET)


"""
Create and train decoding model
"""
n_instances = 1 # number of model instances
# losses = ['l1_pre']
# losses = ['l1_all']
losses = ['l1_pre', 'l1_all']
if USE_CONV:
    losses = [loss + 'conv' for loss in losses]
seeds = [[random.randint(0,10000) for i in range(n_instances)] for j in range(len(losses))]
# for TIMESTEP in [None]:
for TIMESTEP in range(6):
    for loss_ind, loss in enumerate(losses):
        for i in range(0, n_instances):
            print("loss", loss_ind, "instance", i)
            net = RNN.State(activation_func=torch.nn.ReLU(),
                    optimizer=torch.optim.Adam,
                    # lr=8e-5,
                    lr=1e-4,
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    # title="patterns_rev/mscoco_deepgaze3/mscoco_net"+loss+"_"+str(i)+"_excitatory_large_kernels",
                    title=f'patterns_rev/mscoco_deepgaze3/mscoco_net{loss}',
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
            # for layer in [1, 2]:
            layer = 1
            decoding_model = DecodingModelState(torch.optim.Adam,
                                        1e-4,
                                        # f"patterns_rev/mscoco_deepgaze3/decoding_model_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_clipping_1500_current_",
                                        # f"patterns_rev/mscoco_deepgaze3/decoding_model_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_layer{layer}_1500_current_",
                                        f"patterns_rev/mscoco_deepgaze3/decoding_model_timestep_{TIMESTEP}_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_layer{layer}_1500_current_pred{USE_PRED_UNITS}_",
                                        # f"patterns_rev/mscoco_deepgaze3/decoding_model__timestep_{TIMESTEP}predUnits_mscoco_{loss}_{i}_fc_lateral_2layer_{HIDDEN_SIZE}_timesteps_{TIME_STEPS_IMG}_{TIME_STEPS_CORDS}_clipping_1500_current_",
                                        DEVICE,
                                        net,
                                        layer,
                                        HIDDEN_SIZE,
                                        time_step=TIMESTEP,
                                        test_next_fixation=NEXT_FIXATION,
                                        use_pred_units=USE_PRED_UNITS,
                                        test_set=test_set)
            train(decoding_model,
                train_ds=train_set,
                test_ds=test_set,
                loss_fn=loss,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                sequence_length=SEQ_LENGTH,
                verbose=False,
                mnist=MNIST,
                scaler=scaler)
                    
                # save model
            decoding_model.save(NUM_EPOCHS)
