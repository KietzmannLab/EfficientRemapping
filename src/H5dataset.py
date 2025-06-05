import h5py 
import mpl_toolkits.axes_grid1
import numpy as np
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits
import torch
from torchdata import datapipes as dp
import plot
# from IPython.display import  clear_output

def batch_reading_pipe(dataset, chunk_size=128, shuffle_chunks=True, shuffle_buffer_size=10):
    '''
    Create a torchdata pipe for efficient reading from datasets where reads are expensive.
    This is useful for reading from HDF5 files, but can also be used for other file formats.
    Instead of reading one sample at a time, we read a chunk of samples to minimize the number of file reads.

    :param dataset: torch dataset
    :param chunk_size: number of consecutive samples to request in one read operation
    :param shuffle_chunks: whether to shuffle the order of the chunks
    :param shuffle_buffer_size: size of buffer to use for shuffling at the sample level
    :return: torchdata pipe
    '''
    pipe = dp.map.SequenceWrapper(range(len(dataset)))  # indices in dataset
    pipe = dp.iter.Batcher(pipe, batch_size=chunk_size)  # subsequent indices to load together. This minimizes number of file reads
    if shuffle_chunks:
        pipe = dp.iter.Shuffler(pipe, buffer_size=len(pipe))  # shuffle batches so we read from random start points in dataset
    pipe = dp.iter.ShardingFilter(pipe)  # make sure each worker gets its own batches
    pipe = dp.iter.Mapper(pipe, lambda x: dataset[x])  # load data from dataset into memory
    pipe = dp.iter.Mapper(pipe, lambda x: list(zip(*x))) # each datapoint becomes a tuple. For a traditional image, label dataset this is (image, label)
    if shuffle_buffer_size > 0:
        pipe = dp.iter.Shuffler(pipe, buffer_size=shuffle_buffer_size, unbatch_level=1)  # shuffle samples from consecutive batches
    return pipe

class H5dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        dataset_path,
        image_dict = "data",
        fixation_dict = "densenet_deepgaze_fixations",
        classes_from_embeddings = False,
        image_dtype = np.uint8,
        device='cpu',
        use_color=False
    ):
        super().__init__()
        self.split = split
        self._dataset_path = dataset_path
        self.image_dtype = image_dtype
        self.classes_from_embeddings = classes_from_embeddings
        input_data = h5py.File(dataset_path, "r")
        if len(self.split) > 0:
            self.image_label = input_data[self.split]
        else:
            self.image_label = input_data
        self.data = self.image_label[image_dict]

        self.targets = self.image_label[fixation_dict]
        self.device = device
        self.use_color = use_color

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        image = self.data[idx]
        fixation = self.targets[idx]
        return image, fixation
        

    def collate_fn(self, x):
        x = torch.utils.data.default_collate(x)
        x[0] = torch.mean(x[0].float(), dim=3)
        x[0] = x[0] / 255
        x[1] = torch.squeeze(x[1][:, torch.randint(0, 10, (1,1))])
        x[1] = x[1] / 128 - 1
        return x
    
    def collate_fn_color(self, x):
        x = torch.utils.data.default_collate(x)
        x[0] = x[0] / 255
        x[1] = torch.squeeze(x[1][:, torch.randint(0, 10, (1,1))])
        x[1] = x[1] / 128 - 1
        return x
    
    def create_batches(self, batch_size, sequence_length=7, shuffle=True):
        pipe = batch_reading_pipe(self, chunk_size=512, shuffle_chunks=True, shuffle_buffer_size=4096)
        if self.use_color:
            collate_fn = self.collate_fn_color
        else:
            collate_fn = self.collate_fn

        dataloader = torch.utils.data.DataLoader(pipe, batch_size=batch_size, shuffle=shuffle, 
                                                  collate_fn=collate_fn,
                                                  num_workers=2
                                                )
        return dataloader
    

        
def show_image_with_fixations(fileindex, data: H5dataset):
    iterator = iter(data.create_batches(batch_size=1, sequence_length=10, shuffle=False))
    for i in range(1841+1):
        image, xy_coords = next(iterator)
    xy_coords += 1
    xy_coords *= 128
    # image, fixation = data.__getitem__(fileindex)
    # xy_coords = fixation[fixationindex]
    fixation_history_x = xy_coords[:, 0]
    fixation_history_y = xy_coords[:, 1]
    fig, ax = plt.subplots(figsize=(4, 4))
    # ax.plot(fixation_history_x, fixation_history_y, 'o-', color='red')
    cmap = mpl.colormaps['plasma']
    ax.scatter(fixation_history_x, fixation_history_y, 100, cmap='plasma', c=range(7), zorder=100, facecolors='none')
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 6), cmap=cmap), cax=cax, orientation='vertical')
    colors = cmap(np.linspace(0, 1, 7))
    for i in range(6):
        ax.plot(fixation_history_x[i:i+2], fixation_history_y[i:i+2], color=colors[i], zorder=99)
    # ax.imshow(image.mean(axis=2), cmap='gray')
    ax.imshow(image[0], cmap='gray')
    ax.set_axis_off()
    plot.save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/stimulus_generation', bbox_inches='tight')
    print(f'Image: {image.shape}')
    # print(f'Fixation: {fixation.shape}')


if __name__ == '__main__':
        
    h5_dataset = '/share/klab/datasets/optimized_datasets/ms_coco_embeddings_deepgaze.h5'

    # # Each data point has 256,256,3 images, 10,7,2 fixations (10 fixation sequences with seq_len 7)
    # # 2051 data points val
    data_val = H5dataset('val', h5_dataset)
    # # 73000 data points test
    # data_test = H5dataset('test', h5_dataset)
    # # 48236 data points train
    # data_train = H5dataset('train', h5_dataset)

    # Visualizes the first image of the test set with the sequence of fixation coordinates that is used for the Figures
    torch.manual_seed(2553)
    np.random.seed(2553)
    show_image_with_fixations(1058, data_val)
