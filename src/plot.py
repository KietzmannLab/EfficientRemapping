import numpy as np
import scipy.stats
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cycler
from scipy.stats import multivariate_normal
from FovealTransform import FovealTransform
from train import test_batch
import scipy
import matplotlib.colors as mcolor
import pandas as pd
import seaborn as sns
import mpl_toolkits

from Dataset import Dataset
from ModelState import ModelState

# Global matplotlib settings

# colors = cycler('color',
#                 ['#EE6666', '#3388BB', '#9988DD',
#                   '#EECC55', '#88BB44', '#FFBBBB'])


# plt.rc('axes', axisbelow=True, prop_cycle=colors)
# plt.rc('grid', linestyle='--')
# plt.rc('xtick', direction='out', color='black')
# plt.rc('ytick', direction='out', color='black')
# plt.rc('lines', linewidth=2)

# # for a bit nicer font in plots
# mpl.rcParams['font.family'] = ['sans-serif']
# mpl.rcParams['font.size'] = 18

# plt.style.use('ggplot')


# ---------------     Helper/non core functions     ---------------
#

def save_fig(fig, filepath, bbox_inches=None):
    """Convenience wrapper for saving figures in a default "../Results/" directory and auto appends file extensions ".svg"
    and ".png"
    """
    fig.savefig(filepath + ".svg", bbox_inches=bbox_inches, dpi=100)
    fig.savefig(filepath + ".png", bbox_inches=bbox_inches, dpi=100)
    fig.savefig(filepath + ".pdf", bbox_inches=bbox_inches, dpi=100)

def axes_iterator(axes):
    """Iterate over axes. Whether it is a single axis object, a list of axes, or a list of a list of axes
    """
    if isinstance(axes, np.ndarray):
        for ax in axes:
            yield from axes_iterator(ax)
    else:
        yield axes

def init_axes(len_x, figsize, shape=None, colorbar=False):
    """Convenience function for creating subplots with configuratons

    Parameters:
        - len_x: amount of subfigures
        - figsize: size per subplot. Actual figure size depends on the subfigure configuration and if colorbars are visible.
        - shape: subfigure configuration in rows and columns. If 'None', a configuration is chosen to minimise width and height. Default: None
        - colorbar: whether colorbars are going to be used. Used for figsize calculation
    """
    if shape is not None:
        assert isinstance(shape, tuple)
        ncols = shape[0]
        nrows = shape[1]
    else:
        nrows = int(np.sqrt(len_x))
        ncols = int(len_x / nrows)
        while not nrows*ncols == len_x:
            nrows -= 1
            ncols = int(len_x / nrows)

    #figsize = (figsize[1] * ncols + colorbar*0.5*figsize[0], figsize[0] * nrows)
    figsize = (figsize[1] * ncols + 0.5*colorbar*figsize[0], figsize[0] * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize)

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if ax is None: ax=plt.gca()
    if isinstance(ax, np.ndarray):
        ax = ax[0]
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def display(imgs,
            lims=(-1.0, 1.0),
            cmap='seismic',
            size=None,
            figsize=(4,4),
            shape=None,
            colorbar=True,
            axes_visible=True,
            layout='regular',
            figax=None,
            cords=None,
            start_number_cords=0):
    """Convenience function for plotting multiple tensors as images.

    Function to quickly display multiple tensors as images in a grid.
    Image dimensions are expected to be square and are taken to be the square root of the tensor size.
    Tensor dimensions may be arbitrary.
    The images are automatically layed out in a compact grid, but this can be overridden.

    Parameters:
        - imgs: (list of) input tensor(s) (torch.Tensor or numpy.Array)
        - lims: pixel value interval. If 'None', it is set to the highest absolute value in both directions, positive and negative. Default: (-1,1)
        - cmap: color map. Default: 'seismic'
        - size: image width and height. If 'None', it is set to the first round square of the tensor size. Default: None
        - figsize: size per image. Actual figure size depends on the subfigure configuration and if colorbars are visible. Default: (4,4)
        - shape: subfigure configuration, in rows and columns of images. If 'None', a configuration is chosen to minimise width and height. Default: None
        - colorbar: show colorbar for only last row of axes. Default: False
        - axes_visible: show/hide axes. Default: True
        - layout: matplotlib layout. Default: 'regular'
        - figax: if not 'None', use existing figure and axes object. Default: None
        -cmaps: pass list of colormaps 
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
        shape = (1,1)

    if size is not None:
        if not isinstance(size, tuple):
            size = (size, size)

    # convert to numpy if not already so
    imgs = [im.detach().cpu().numpy() if isinstance(im, torch.Tensor) else im for im in imgs]

    if lims is None:
        mx = max([max(im.max(),abs(im.min())) for im in imgs])
        lims = (-mx, mx)
    if figax is None:
        fig, axes = init_axes(len(imgs), figsize, shape=shape, colorbar=colorbar)
    else:
        fig, axes = figax
  
    for i, ax in enumerate(axes_iterator(axes)):
       
        img = imgs[i]
        ax.grid(visible=False)
        if size is None:
            _size = int(np.sqrt(img.size))
            img = img[:_size*_size].reshape(_size,_size)
        else:
            img = img[:size[0]*size[1]].reshape(size[0],size[1])
        plot_im = ax.imshow(img, cmap=cmap)
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(color='black')

        if cords is not None and shape[0] != 10 and i%shape[0] >= shape[0] - start_number_cords:
            dx = min(max(cords[i][0]*128, -64), 63)
            dy = min(max(cords[i][1]*128, -64), 63)
            ax.arrow(64, 64, dx, dy, color='green', head_width=5, length_includes_head=True)
        
        if shape[0] == 10 and cords is not None and i < 10:
            dx = min(max(cords[i][0]*54, -27), 26)
            dy = min(max(cords[i][1]*54, -27), 26)
            ax.arrow(27, 27, dx, dy, color='green', head_width=5, length_includes_head=True)

        ax.label_outer()

        if lims is not None:
            plot_im.set_clim(lims[0], lims[1])

        if axes_visible == False:
            ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)

  
    if colorbar:

        if isinstance(axes, np.ndarray):
            fig.colorbar(plot_im, ax=axes, shrink=0.50)
            set_size(figsize[0]+0.5, figsize[1], axes[-1])
        else:
            fig.colorbar(plot_im, ax=axes)
            set_size(figsize[0]+0.5, figsize[1], axes)

    return fig, axes



def scatter(x, y, discrete=False, figsize=(8,6), color='r', xlabel="", ylabel="", legend=None, figax=None, alpha=1):
    """Convenience function to create scatter plots

    Parameters:
        - x: x data points. Array or list of arrays.
        - y: y data points
        - discrete: whether xaxis ticks should be integer values. Default: False
        - figsize: matplotlib figsize. Default: (8,6)
        - xlabel: Default: ""
        - ylabel: Default: ""
        - legend: display legend. Default: None
        - figax: if not 'None', use existing figure and axes objects. Default: None
    """
    if not isinstance(color,str) and not isinstance(color, np.ndarray):
        color = list(mcolor.TABLEAU_COLORS)[color]
    if figax is None:
        fig, axes = plt.subplots(1, figsize=figsize)
    else:
        fig, axes = figax

    if isinstance(x, list):
        ax = axes
        for i, _x in enumerate(x):
            ax.scatter(_x, y[i], c=color, alpha=alpha)
    else:
        axes.scatter(x, y, c=color, alpha=alpha)

    if discrete:
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid();

    if legend is not None:
        axes.legend(legend)

    return fig, axes


def linePlot(x, y, discrete=False, figsize=(8,6), color = 'r', linestyle='-', xlabel="", ylabel="", legend=None, label=None, figax=None, ylim=None):
    """Convenience function to create line plots

    Parameters:
        - x: x data points. Array or list of arrays.
        - y: y data points
        - discrete: whether xaxis ticks should be integer values. Default: False
        - figsize: matplotlib figsize. Default: (8,6)
        - color: list of class indices that should have the same color
        - xlabel: Default: ""
        - ylabel: Default: ""
        - legend: display legend. Default: None
        - figax: if not 'None', use existing figure and axes objects. Default: None
    """
    if figax is None:
        fig, axes = plt.subplots(1, figsize=figsize)
    else:
        fig, axes = figax

    if isinstance(x, list):
        ax = axes
        for i, _x in enumerate(x):
            if isinstance(color, list) or isinstance(color, range):
                c = color[i]
            else:
                c = color
            if isinstance(linestyle, list) or isinstance(linestyle, range):
                line = linestyle[i]
            else:
                line = linestyle
            if label is not None:
                ax.plot(_x, y[i], c=list(mcolor.TABLEAU_COLORS)[c], label=label[i], linestyle=line)
            else:
                ax.plot(_x, y[i], c=list(mcolor.TABLEAU_COLORS)[c], linestyle=line)
    else:
        if label is not None:
            axes.plot(x, y, c=list(mcolor.TABLEAU_COLORS)[color], label=label, linestyle=linestyle)
        else:
            axes.plot(x, y, c=list(mcolor.TABLEAU_COLORS)[color], linestyle=linestyle)

    if discrete:
        axes.xaxis.set_major_locator(MaxNLocator(integer=True));
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid()
    if ylim is not None:
        axes.set_ylim(bottom=ylim[0], top=ylim[1])

    if legend is not None:
        axes.legend(legend)
    if label is not None:
        axes.legend()

    return fig, axes

def decoding_results():
    sns.set_context('talk')
    F_PATH = 'EmergentPredictiveCoding/Results/Fig2_mscoco/'
    results = pd.read_csv(F_PATH+"decoder_preds.csv")
    # fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    # ax[0].scatter(results['x_pred'], results['x_target'], alpha=0.5)
    # ax[1].scatter(results['y_pred'], results['y_target'], alpha=0.5)
    # save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/decoder_predictions')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5), sharex=True)

    # sns.scatterplot(data=results, x='x_target', y='x_pred', 
    #             # palette=pallette_greys_non_lesioned,
    #             ax=ax1, edgecolor='black', alpha=0.25, s=20)
    sns.kdeplot(data=results, x='x_target', y='x_pred', 
                # palette=pallette_greys_non_lesioned,
                ax=ax1, edgecolor='black', fill=True, cmap='plasma')
    # ax1.plot((-1, 1), (-1, 1), color='black')
    # ax1.xaxis.set_tick_params(labelbottom=False)
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    # ax1.set_title('decoding results x')
    # ax1.set_xlabel('target [x position]')
    ax1.set_ylabel('predicted')
    ax1.text(-1, 1, "R² = 0.91")
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    # cax.set_ylabel('density', rotation=-90, va='bottom')
    fig.colorbar(mpl.cm.ScalarMappable(cmap='plasma'), cax=cax, orientation='vertical')


    # sns.scatterplot(data=results, x='y_target', y='y_pred', 
    #             # palette=pallette_greys_non_lesioned,
    #             ax=ax2, edgecolor='black', alpha=0.25, s=20)
    sns.kdeplot(data=results, x='y_target', y='y_pred', 
                # palette=pallette_greys_non_lesioned,
                ax=ax2, edgecolor='black', fill=True, cmap='plasma')
    # ax2.plot((-1, 1), (-1, 1), color='black')
    ax2.set_xticks([-1, 0, 1], ['-1', '0', '1'])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    # ax2.xaxis.set_tick_params(labelbottom=True)
    # ax2.set_title('decoding results y')
    ax2.set_xlabel('target')
    ax2.set_ylabel('predicted')
    ax2.text(-1, 1, "R² = 0.93")
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    # cax.set_ylabel('density', rotation=-90, va='bottom')
    fig.colorbar(mpl.cm.ScalarMappable(cmap='plasma'), cax=cax, orientation='vertical')

    fig.tight_layout()
    save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/decoder_predictions')
    sns.set_context(None)

def decoding_example():
    sns.set_context('talk')
    # F_PATH = 'EmergentPredictiveCoding/Results/Fig2_mscoco/'
    # results = pd.read_csv(F_PATH+"decoder_preds.csv")
    # fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    # ax[0].scatter(results['x_pred'], results['x_target'], alpha=0.5)
    # ax[1].scatter(results['y_pred'], results['y_target'], alpha=0.5)
    # save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/decoder_predictions')
    F_PATH = 'EmergentPredictiveCoding/Results/Fig2_mscoco/'
    results = pd.read_csv(F_PATH+"example_preds.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    ax1.scatter(range(42), results['x_pred'], label='decoded x', c=mpl.colormaps['plasma'](np.arange(0, 1, 0.1))[3])
    ax1.plot(range(42), results['x_target'], label='target', c='darkgray')
    # ax1.plot((-1, 1), (-1, 1), color='black')
    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    # ax1.set_title('example decoding x')
    # ax1.set_xlabel('time [model time steps]')
    ax1.set_ylabel('[x]')
    # ax1.legend()

    ax2.scatter(range(42), results['y_pred'], label='decoded y', c=mpl.colormaps['plasma'](np.arange(0, 1, 0.1))[5])
    ax2.plot(range(42), results['y_target'], label='target', c='darkgray')
    # ax1.plot((-1, 1), (-1, 1), color='black')
    ax2.xaxis.set_tick_params(labelbottom=True)
    # print(ax2.get_xticklabels())
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    # ax2.set_title('example decoding y')
    ax2.set_xlabel('time [model time steps]')
    ax2.set_ylabel('[y]')
    fig.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/decoder_example')
    sns.set_context(None)

def training_progress(net:ModelState, save=True):
    """
    
    wrapper function that shows model training

    """
    fig, axes = init_axes(1, figsize=(6,8))


    axes.plot(np.arange(1, len(net.results["train loss"])+1), net.results["train loss"], label="Training set")
    axes.plot(np.arange(1, len(net.results["test loss"])+1), net.results["test loss"], label="Test set")

   
   
  
    axes.xaxis.set_major_locator(MaxNLocator(integer=True));
    axes.set_xlabel("Training time",fontsize=16)
    axes.set_ylabel("Loss",fontsize=16)
    axes.legend()
    axes.set_title('Loss network', fontsize=18)
    axes.grid(True)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    fig.tight_layout()

    if save is True:
        save_fig(fig, "training-progress", bbox_inches='tight')



#
# ---------------     Plotting code for figures paper     ---------------
#
def example_sequence_state(net:ModelState, dataset:Dataset, latent=False, seed=2553, save=False, mnist=False, use_conv=False, warp_imgs=True):
    """
    visualises input and internal drive for a sample sequence
    """
    device = 'cuda'
    if seed != None:
        torch.manual_seed(seed)
        # np.random.seed(seed)
    if mnist:
        batches, fixations = dataset.create_list_batches(batch_size=-1, sequence_length=10, shuffle=False)
        ex_seq = batches[0,:,0]
        fixations = torch.ones((10, 2)) * 14
        fixations[5:, 0] += 28
        for i in range(5):
            fixations[i, 1] += i * 28
        for i in range(5):
            fixations[9-i, 1] += i * 28

        fixations[:, 0] = fixations[:, 0] / 28 - 1
        fixations[:, 1] = fixations[:, 1] / 70 - 1
        foveal_transform = net.foveal_transform
        ex_seq = ex_seq.reshape(1, 1, 140, 56)
        fixations = fixations.reshape(1, 10, 2)
    else:
        iterator = iter(dataset.create_batches(batch_size=1, sequence_length=10, shuffle=False))
        for i in range(1841+1):
            batches, fixations = next(iterator)
        # next(iterator)
        # batches, fixations = next(iterator)
        # batches = batches[818]
        # fixations = fixations[818]
        batches = batches.reshape(1, 1, 256, 256)
        fixations = fixations.reshape(1, 7, 2)
        foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
                                                jitter_amount=0, device=device, warp_imgs=warp_imgs)
    batches = batches.to(device)
    fixations = fixations.to(device)
    if mnist:
        shape = ex_seq.shape
        ex_seq = foveal_transform(ex_seq, fixations)[0]
    else:
        ex_seq = foveal_transform(batches, fixations)[0]
    ex_seq = ex_seq.to(device)

    
    if mnist:
        # Make fixations egocentric
        for i in range(fixations.shape[1]):
            if i == fixations.shape[1]-1:
                fixations[:, i] = fixations[:, i] - fixations[:, i]
            else:
                fixations[:, i] = fixations[:, i+1] - fixations[:, i]
        fixations[:, :, 1] = fixations[:, :, 1] / 0.4
    else:
        if not net.model.use_grid_coding:
            # Make fixations egocentric
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]

    input_size = ex_seq.shape[-1] # make sure we only visualize input units and no latent resources
    X = []; P = []; H=[]; T=[]; L=[]; C=[]
    
    h = net.model.init_state(1)
    recurrent_state_p = None
    recurrent_state = None
    # fig, ax = plt.subplots()
    with torch.no_grad():
        for i, x in enumerate(ex_seq):
            x = x.reshape(1, -1)
            if use_conv:
                x = x.reshape(x.shape[0], 128, 128)
            for t in range(net.model.time_steps_img):
                if t >=  net.model.time_steps_img - net.model.time_steps_cords:
                    h, l_a, recurrent_state = net.model(x, fixations[0, i].reshape(1,2), state=h, recurrent_state=recurrent_state)
                    p, recurrent_state_p = net.predict(x, fixations[0, i].reshape(1,2), recurrent_state_p)
                else:
                    h, l_a, recurrent_state = net.model(x, torch.zeros_like(fixations[0, i].reshape(1,2)), state=h, recurrent_state=recurrent_state)
                    p, recurrent_state_p = net.predict(x, torch.zeros_like(fixations[0, i].reshape(1,2)), recurrent_state_p)
                X.append(x[0,:input_size].detach().cpu())
                P.append(p.detach().cpu())
                H.append(None)
                T.append(l_a[0][0,:input_size].detach().cpu())
                C.append(fixations[0, i].cpu())

    return X, P, H, T, C

def get_image_performance(net:ModelState, dataset:Dataset, seed=2553):
    """
    visualises input and internal drive for a sample sequence
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)


    loader = dataset.create_batches(batch_size=1, sequence_length=10, shuffle=False)
    image_stats = torch.zeros(2051, 3)
    i = 0
    for batches, fixations in loader:
        # batches = batches.reshape(1, 1, 256, 256)
        # foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
        #                                         jitter_amount=0, device='cuda', warp_imgs=False)
        # batches = batches.to('cuda')
        # fixations = fixations.to('cuda')
        # ex_seq = foveal_transform(batches, fixations)[0]
        # ex_seq = ex_seq.to('cuda')
        fixations = fixations.reshape(1, 7, 2)
        loss, _, _ = net.run(batches, fixations, 'l1_all', None)
        image_stats[i, 0] = i
        image_stats[i, 1] = loss
        image_stats[i, 2] = torch.std(batches, dim=None)
        i += 1
    
    return image_stats


def compare_random_fixations(net:ModelState, dataset:Dataset, latent=False, seed=2553, save=False, loss_fn='l1_all', use_conv=False, warp_imgs=True, include_initial=True, use_resNet=False, feature_size=128*128, mnist=False, just_onset=False, return_fixations=False):
    """
    tests model against model with shuffled fixation order
    """
    # losses_baseline = np.empty((0,0))
    # losses = np.empty((0,0))
    losses_baseline = None
    losses = None
    total_fixations = None
    for test in range(2):
        torch.manual_seed(seed)
        tot_loss = 0
        state = None
        batch_size = 1024
        if mnist:
            batches, fixations = dataset.create_list_batches(batch_size=batch_size, sequence_length=10, shuffle=False)
            loader = [(batches[i], fixations[i]) for i in range(batches.shape[0])]
        else:
            loader = dataset.create_batches(batch_size=batch_size, shuffle=False)
        if mnist:
            foveal_transform = net.foveal_transform
        else:
            foveal_transform = net.foveal_transform
            # foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
            #                                     jitter_amount=0, device='cuda', warp_imgs=warp_imgs)
        counter = 0
        for batch, fixations in loader:
            if test == 0 and total_fixations is None:
                total_fixations = fixations.clone()
            elif test == 0:
                total_fixations = torch.cat((total_fixations, fixations), dim=0)
            recurrent_state_pred = None
            batch = batch.to('cuda')
            fixations = fixations.to('cuda')
            if use_resNet:
                seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
            else:
                if mnist:
                    batch = batch.permute(1, 0)
                    fixations = fixations.permute(1, 0)
                    fixations = fixations.reshape(-1, 10, 2)
                    seq = foveal_transform(batch.reshape(batch.shape[0], 1, 140, 56), fixations)
                else:
                    seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
            seq = seq.to('cuda')
            if test == 1:
                fixations = fixations[:, torch.randperm(fixations.shape[1])]
            if not net.model.use_grid_coding:
                for i in range(fixations.shape[1]):
                    if i == fixations.shape[1]-1:
                        fixations[:, i] = fixations[:, i] - fixations[:, i]
                    else:
                        fixations[:, i] = fixations[:, i+1] - fixations[:, i]
            with torch.no_grad():
                losses_batch = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
                for i in range(seq.shape[1]):
                    image = seq[:, i]
                    if use_conv:
                        image = image.reshape(image.shape[0], 128, 128)
                    for t in range(net.model.time_steps_img):
                        if t >= net.model.time_steps_img - net.model.time_steps_cords:
                            pred, recurrent_state_pred = net.predict(image, fixations[:, i], recurrent_state_pred)
                        else:
                            pred, recurrent_state_pred = net.predict(image, torch.zeros_like(fixations[:, i]), recurrent_state_pred)
                        if i != 0 or t != 0 or include_initial:
                            if t == 0 or not just_onset:
                                if not use_resNet:
                                    tot_loss += torch.mean(torch.abs(pred[:, :image.shape[1]] + image)) * batch.shape[0]
                                    if test == 1:
                                        # losses_baseline = np.append(losses_baseline, torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=1).cpu().numpy())
                                        losses_batch[:, i, t] = torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=1)
                                    else:
                                        # losses = np.append(losses, torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=1).cpu().numpy())
                                        losses_batch[:, i, t] = torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=1)
                                else:
                                    tot_loss += torch.mean(torch.abs(pred + net.model.flatten(net.model.resNet(seq[:, i])))) * batch.shape[0]
                if test == 1:
                    if losses_baseline is None:
                        losses_baseline = losses_batch.cpu().numpy()
                    else:
                        losses_baseline = np.concatenate((losses_baseline, losses_batch.cpu().numpy()), axis=0)
                else:
                    if losses is None:
                        losses = losses_batch.cpu().numpy()
                    else:
                        losses = np.concatenate((losses, losses_batch.cpu().numpy()), axis=0)
            if not just_onset:
                counter += batch.shape[0] * net.model.time_steps_img
            else:
                counter += batch.shape[0]
        tot_loss /= counter
        if not just_onset:
            if test == 0:
                print("Normal fixation order")
            else:
                print("Shuffled fixation order")
            print("Test loss:     {:.8f}".format(tot_loss))
        else:
            if test == 0:
                print("Normal fixation order only onset")
            else:
                print("Shuffled fixation order only onset")
            print("Test loss:     {:.8f}".format(tot_loss))

    if return_fixations:
        return losses, losses_baseline, total_fixations
    return losses, losses_baseline

def compare_average_img(net:ModelState, dataset:Dataset, trainset:Dataset, latent=False, seed=2553, save=False, use_conv=False, warp_imgs=True, include_initial=True, use_resNet=False, feature_size=128*128, return_feedback=False):
    """
    tests model against average image of test set
    """
    device = 'cuda'
    tot_loss = 0
    tot_loss_baseline = 0
    batch_size = 1024
    loader = trainset.create_batches(batch_size=batch_size, shuffle=False)
    if return_feedback:
        feedback = None
    foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
                                                jitter_amount=0, device=device, warp_imgs=warp_imgs)
    if not use_resNet:
        avg_img = torch.zeros((128 * 128))
    else:
        avg_img = torch.zeros(feature_size).to(device)
    counter = 0
    torch.manual_seed(seed)
    for batch, fixations in loader:
        if use_resNet:
            batch = batch.to(device)
            fixations = fixations.to(device)
        if use_resNet:
            seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
            seq = seq.to(device)
        else:
            seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
        for i in range(seq.shape[1]):
            if not use_resNet:
                avg_img += torch.mean(seq[:, i], dim=0) * batch.shape[0]
            else:
                features = net.model.resNet(seq[:, i])
                features = net.model.flatten(features)
                avg_img += torch.mean(features, dim=0) * batch.shape[0]
            counter += batch.shape[0]
    avg_img /= counter
    if not use_resNet:
        fig, _ = display(avg_img, cmap='gray', lims=(avg_img.min(), avg_img.max()), shape=(1,1), figsize=(6,6), axes_visible=False, layout='tight')
        save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/avg_img', bbox_inches='tight')
        avg_img_z_score = scipy.stats.zscore(avg_img.numpy(), axis=None)
        fig, _ = display(avg_img_z_score, lims=None, shape=(1,1), figsize=(6,6), axes_visible=False, layout='tight')
        save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/avg_img_z_score', bbox_inches='tight')
    avg_img = avg_img.to(device)
    counter = 0
    # losses = np.empty((0,0))
    # losses_baseline = np.empty((0,0))
    losses = None
    losses_baseline = None
    loader = dataset.create_batches(batch_size=batch_size, shuffle=False)
    torch.manual_seed(seed)
    batch_nr = -1
    for batch, fixations in loader:
        batch_nr += 1
        recurrent_state_pred = None
        batch = batch.to(device)
        fixations = fixations.to(device)
        if use_resNet:
            seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
        else:
            seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
        seq = seq.to(device)
        if not net.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
        with torch.no_grad():
            losses_batch = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
            losses_batch_baseline = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
            for i in range(seq.shape[1]):
                image = seq[:, i]
                if use_conv:
                    image = image.reshape(image.shape[0], 128, 128)
                for t in range(net.model.time_steps_img):
                    if t >= net.model.time_steps_img - net.model.time_steps_cords:
                        pred, recurrent_state_pred = net.predict(image, fixations[:, i], recurrent_state_pred)
                    else:
                        pred, recurrent_state_pred = net.predict(image, torch.zeros_like(fixations[:, i]), recurrent_state_pred)
                    if i != 0 or t != 0 or include_initial:
                        # print(i, torch.mean(torch.abs(pred + image)).item(), torch.mean(pred).item(), torch.mean(image).item())
                        if not use_resNet:
                            tot_loss_baseline += torch.mean(torch.abs(seq[:, i] - avg_img)) * batch.shape[0]
                            tot_loss += torch.mean(torch.abs(pred[:, :image.shape[1]] + image)) * batch.shape[0]
                            losses_batch_baseline[:, i, t] = torch.mean(torch.abs(seq[:, i] - avg_img), dim=1)
                            losses_batch[:, i, t] = torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=1)
                        else:
                            tot_loss += torch.mean(torch.abs(pred + net.model.flatten(net.model.resNet(seq[:, i])))) * batch.shape[0]
                            tot_loss_baseline += torch.mean(torch.abs(net.model.flatten(net.model.resNet(seq[:, i])) - avg_img)) * batch.shape[0]
                        if return_feedback:
                            if feedback is None:
                                feedback = pred
                            else:
                                feedback = torch.cat((feedback, pred), dim=0)
            if losses_baseline is None:
                losses_baseline = losses_batch_baseline
            else:
                losses_baseline = np.concatenate((losses_baseline, losses_batch_baseline.cpu().numpy()), axis=0)
            if losses is None:
                losses = losses_batch
            else:
                losses = np.concatenate((losses, losses_batch.cpu().numpy()), axis=0)
        counter += batch.shape[0] * net.model.time_steps_img

    tot_loss /= counter
    tot_loss_baseline /= counter
    if use_resNet:
        if include_initial:
            print("Test loss first layer including initial pred:     {:.8f}".format(tot_loss))
            print("Test loss average feature vector test set including initial pred:     {:.8f}".format(tot_loss_baseline))
        else:
            print("Test loss first layer without initial pred:     {:.8f}".format(tot_loss * 7/6))
            print("Test loss average feature vector test set without initial pred:     {:.8f}".format(tot_loss_baseline * 7/6))
    else:
        if include_initial:
            print("Test loss first layer including initial pred:     {:.8f}".format(tot_loss))
            print("Test loss average image test set including initial pred:     {:.8f}".format(tot_loss_baseline))
        else:
            print("Test loss first layer without initial pred:     {:.8f}".format(tot_loss * 7/6))
            print("Test loss average image test set without initial pred:     {:.8f}".format(tot_loss_baseline * 7/6))
    if return_feedback:
        return losses, losses_baseline, feedback
    return losses, losses_baseline

def compare_avg_local_crop(net:ModelState, dataset:Dataset, trainset:Dataset, latent=False, seed=2553, save=False, use_conv=False, warp_imgs=True, include_initial=True, use_resNet=False, feature_size=128*128, return_feedback=False):
    """
    tests model against average image of test set
    """
    device = 'cuda'
    tot_loss = 0
    tot_loss_baseline = 0
    batch_size = 1024
    loader = trainset.create_batches(batch_size=batch_size, shuffle=False)
    if return_feedback:
        feedback = None
    foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
                                                jitter_amount=0, device=device, warp_imgs=warp_imgs)
    if not use_resNet:
        avg_img = torch.zeros((256, 256))
    else:
        avg_img = torch.zeros(feature_size).to(device)
    counter = 0
    torch.manual_seed(seed)
    for batch, fixations in loader:
        print(batch.shape)
        avg_img += torch.mean(batch, dim=0) * batch.shape[0]
        counter += batch.shape[0]
    avg_img /= counter
    if not use_resNet:
        fig, _ = display(avg_img, cmap='gray', lims=(avg_img.min(), avg_img.max()), shape=(1,1), figsize=(6,6), axes_visible=False, layout='tight')
        save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/avg_img_global', bbox_inches='tight')
    avg_img = torch.unsqueeze(avg_img, 0)
    counter = 0
    # losses = np.empty((0,0))
    # losses_baseline = np.empty((0,0))
    losses = None
    losses_baseline = None
    loader = dataset.create_batches(batch_size=batch_size, shuffle=False)
    torch.manual_seed(seed)
    batch_nr = -1
    for batch, fixations in loader:
        avg_img_batch = avg_img.repeat(batch.shape[0], 1, 1)
        avg_img_batch = avg_img_batch.to(device)
        batch_nr += 1
        recurrent_state_pred = None
        batch = batch.to(device)
        fixations = fixations.to(device)
        if use_resNet:
            seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
        else:
            seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
            control_seq = foveal_transform(avg_img_batch.reshape(avg_img_batch.shape[0], 1, 256, 256), fixations)
        seq = seq.to(device)
        control_seq = control_seq.to(device)
        if not net.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
        with torch.no_grad():
            losses_batch = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
            losses_batch_baseline = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
            for i in range(seq.shape[1]):
                image = seq[:, i]
                control_img = control_seq[:, i]
                if use_conv:
                    image = image.reshape(image.shape[0], 128, 128)
                for t in range(net.model.time_steps_img):
                    if t >= net.model.time_steps_img - net.model.time_steps_cords:
                        pred, recurrent_state_pred = net.predict(image, fixations[:, i], recurrent_state_pred)
                    else:
                        pred, recurrent_state_pred = net.predict(image, torch.zeros_like(fixations[:, i]), recurrent_state_pred)
                    if i != 0 or t != 0 or include_initial:
                        # print(i, torch.mean(torch.abs(pred + image)).item(), torch.mean(pred).item(), torch.mean(image).item())
                        if not use_resNet:
                            tot_loss_baseline += torch.mean(torch.abs(seq[:, i] - control_img)) * batch.shape[0]
                            tot_loss += torch.mean(torch.abs(pred[:, :image.shape[1]] + image)) * batch.shape[0]
                            losses_batch_baseline[:, i, t] = torch.mean(torch.abs(seq[:, i] - control_img), dim=1)
                            losses_batch[:, i, t] = torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=1)
                        else:
                            tot_loss += torch.mean(torch.abs(pred + net.model.flatten(net.model.resNet(seq[:, i])))) * batch.shape[0]
                            tot_loss_baseline += torch.mean(torch.abs(net.model.flatten(net.model.resNet(seq[:, i])) - avg_img)) * batch.shape[0]
                        if return_feedback:
                            if feedback is None:
                                feedback = pred
                            else:
                                feedback = torch.cat((feedback, pred), dim=0)
            if losses_baseline is None:
                losses_baseline = losses_batch_baseline
            else:
                losses_baseline = np.concatenate((losses_baseline, losses_batch_baseline.cpu().numpy()), axis=0)
            if losses is None:
                losses = losses_batch
            else:
                losses = np.concatenate((losses, losses_batch.cpu().numpy()), axis=0)
        counter += batch.shape[0] * net.model.time_steps_img

    tot_loss /= counter
    tot_loss_baseline /= counter
    if use_resNet:
        if include_initial:
            print("Test loss first layer including initial pred:     {:.8f}".format(tot_loss))
            print("Test loss average feature vector test set including initial pred:     {:.8f}".format(tot_loss_baseline))
        else:
            print("Test loss first layer without initial pred:     {:.8f}".format(tot_loss * 7/6))
            print("Test loss average feature vector test set without initial pred:     {:.8f}".format(tot_loss_baseline * 7/6))
    else:
        if include_initial:
            print("Test loss first layer including initial pred:     {:.8f}".format(tot_loss))
            print("Test loss average local crop including initial pred:     {:.8f}".format(tot_loss_baseline))
        else:
            print("Test loss first layer without initial pred:     {:.8f}".format(tot_loss * 7/6))
            print("Test loss average local crop without initial pred:     {:.8f}".format(tot_loss_baseline * 7/6))
    if return_feedback:
        return losses, losses_baseline, feedback
    return losses, losses_baseline

def compare_average_color(net:ModelState, dataset:Dataset, trainset:Dataset, latent=False, seed=2553, save=False, use_conv=False, warp_imgs=True, include_initial=True, use_resNet=False):
    """
    tests model against average color of test set
    """
    tot_loss = 0
    tot_loss_baseline = 0
    batch_size = 1024
    loader = trainset.create_batches(batch_size=batch_size, shuffle=False)
    foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
                                                jitter_amount=0, device='cuda', warp_imgs=warp_imgs)
    avg_color = 0
    counter = 0
    # losses_baseline = np.zeros((0,0))
    losses_baseline = None
    torch.manual_seed(seed)
    for batch, fixations in loader:
        if use_resNet:
            batch = batch.to('cuda')
            fixations = fixations.to('cuda')
            seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
            seq = seq.to('cuda')
        else:
            seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
        for i in range(seq.shape[1]):
            if not use_resNet:
                avg_color += torch.mean(seq[:, i]) * batch.shape[0]
            else:
                avg_color += torch.mean(net.model.resNet(seq[:,i])) * batch.shape[0]
            counter += batch.shape[0]
    avg_color /= counter
    print(f"Avg color: {avg_color}")
    avg_color = avg_color.to('cuda')
    counter = 0
    loader = dataset.create_batches(batch_size=batch_size, shuffle=False)
    torch.manual_seed(seed)
    for batch, fixations in loader:
        recurrent_state_pred = None
        batch = batch.to('cuda')
        fixations = fixations.to('cuda')
        if use_resNet:
            seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
        else:
            seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
        seq = seq.to('cuda')
        if not net.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
        with torch.no_grad():
            losses_batch_baseline = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
            for i in range(seq.shape[1]):
                image = seq[:, i]
                if use_conv:
                    image = image.reshape(image.shape[0], 128, 128)
                for t in range(net.model.time_steps_img):
                    if t >= net.model.time_steps_img - net.model.time_steps_cords:
                        pred, recurrent_state_pred = net.predict(image, fixations[:, i], recurrent_state_pred)
                    else:
                        pred, recurrent_state_pred = net.predict(image, torch.zeros_like(fixations[:, i]), recurrent_state_pred)
                    if i != 0 or t != 0 or include_initial:
                        # TODO remove the :image.shape[1]
                        if not use_resNet:
                            tot_loss += torch.mean(torch.abs(pred[:, :image.shape[1]] + image)) * batch.shape[0]
                            tot_loss_baseline += torch.mean(torch.abs(seq[:, i] - avg_color)) * batch.shape[0]
                            losses_batch_baseline[:, i, t] = torch.mean(torch.abs(seq[:, i] - avg_color), dim=1)
                        else:
                            tot_loss += torch.mean(torch.abs(pred + net.model.flatten(net.model.resNet(image)))) * batch.shape[0]
                            tot_loss_baseline += torch.mean(torch.abs(net.model.flatten(net.model.resNet(seq[:, i])) - avg_color)) * batch.shape[0]
            if losses_baseline is None:
                losses_baseline = losses_batch_baseline
            else:
                losses_baseline = np.concatenate((losses_baseline, losses_batch_baseline.cpu().numpy()), axis=0)
        counter += batch.shape[0] * net.model.time_steps_img

    tot_loss /= counter
    tot_loss_baseline /= counter
    if use_resNet:
        if include_initial:
            print("Test loss first layer including intial pred:     {:.8f}".format(tot_loss))
            print("Test loss average feature value test set including initial pred:     {:.8f}".format(tot_loss_baseline))
        else:
            print("Test loss first layer without initial pred:     {:.8f}".format(tot_loss * 7/6))
            print("Test loss average feature value test set without initial pred:     {:.8f}".format(tot_loss_baseline * 7/6))
    else:
        if include_initial:
            print("Test loss first layer including intial pred:     {:.8f}".format(tot_loss))
            print("Test loss average color test set including initial pred:     {:.8f}".format(tot_loss_baseline))
        else:
            print("Test loss first layer without initial pred:     {:.8f}".format(tot_loss * 7/6))
            print("Test loss average color test set without initial pred:     {:.8f}".format(tot_loss_baseline * 7/6))
    return losses_baseline

def compare_previous_fixation(net:ModelState, dataset:Dataset, latent=False, seed=2553, save=False, use_conv=False, warp_imgs=True, use_resNet=False, just_fixation_onset=False, returnFeedback=False):
    """
    tests model against previous fixation
    """
    device = 'cuda'
    tot_loss = 0
    tot_loss_baseline = 0
    batch_size = 1024
    loader = dataset.create_batches(batch_size=batch_size, shuffle=False)
    foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
                                                jitter_amount=0, device=device, warp_imgs=warp_imgs)
    counter = 0
    # losses_baseline = np.empty((0,0))
    # losses = np.empty((0,0))
    losses_baseline = None
    losses = None
    torch.manual_seed(seed)
    batch_nr = -1
    feedback_grouped = torch.zeros((2051, 2, 7, 6, 128*128))
    for batch, fixations in loader:
        batch_nr += 1
        recurrent_state_pred = None
        batch = batch.to(device)
        fixations = fixations.to(device)
        if use_resNet:
            seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
        else:
            seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
        seq = seq.to(device)
        prev_image = None
        if not net.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
        with torch.no_grad():
            losses_batch = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
            losses_batch_baseline = torch.zeros((batch.shape[0], seq.shape[1], net.model.time_steps_img))
            for i in range(seq.shape[1]):
                image = seq[:, i]
                if use_conv:
                    image = image.reshape(image.shape[0], 128, 128)
                for t in range(net.model.time_steps_img):
                    if t >= net.model.time_steps_img - net.model.time_steps_cords:
                        pred, recurrent_state_pred = net.predict(image, fixations[:, i], recurrent_state_pred)
                    else:
                        pred, recurrent_state_pred = net.predict(image, torch.zeros_like(fixations[:, i]), recurrent_state_pred)
                    if i != 0:
                        if t == 0 or not just_fixation_onset:
                            # TODO remove the :image.shape[1]
                            if not use_resNet:
                                tot_loss += torch.mean(torch.abs(pred[:, :image.shape[1]] + image)) * batch.shape[0]
                                tot_loss_baseline += torch.mean(torch.abs(image - prev_image)) * batch.shape[0]
                                losses_batch_baseline[:, i, t] = torch.mean(torch.abs(image - prev_image), dim=1)
                                losses_batch[:, i, t] = torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=1)
                            else:
                                tot_loss += torch.mean(torch.abs(pred + net.model.flatten(net.model.resNet(image)))) * batch.shape[0]
                                tot_loss_baseline += torch.mean(torch.abs(net.model.flatten(net.model.resNet(seq[:, i])) - net.model.flatten(net.model.resNet(prev_image)))) * batch.shape[0]
                    if returnFeedback:
                        if batch_nr < 2:
                            feedback_grouped[batch_nr*1024:(batch_nr+1)*1024, 0, i, t] = pred
                            feedback_grouped[batch_nr*1024:(batch_nr+1)*1024, 1, i, t] = image * (-1)
                        else:
                            feedback_grouped[batch_nr*1024:, 0, i, t] = pred
                            feedback_grouped[batch_nr*1024:, 1, i, t] = image * (-1)
                prev_image = image
            if losses_baseline is None:
                losses_baseline = losses_batch_baseline
            else:
                losses_baseline = np.concatenate((losses_baseline, losses_batch_baseline.cpu().numpy()), axis=0)
            if losses is None:
                losses = losses_batch
            else:
                losses = np.concatenate((losses, losses_batch.cpu().numpy()), axis=0)
        if just_fixation_onset:
            counter += batch.shape[0]
        else:
            counter += batch.shape[0] * net.model.time_steps_img


    tot_loss /= counter
    tot_loss_baseline /= counter
    if just_fixation_onset:
        print("Test loss first layer fixation onset:     {:.8f}".format(tot_loss* 7/6))
        print("Test loss directly after saccade compared against previous fixation:     {:.8f}".format(tot_loss_baseline * 7/6))
    else:
        print("Test loss first layer:     {:.8f}".format(tot_loss* 7/6))
        print("Test loss compared against previous fixation:     {:.8f}".format(tot_loss_baseline * 7/6))
    if returnFeedback:
        return losses, losses_baseline, feedback_grouped
    return losses, losses_baseline

def extract_loss_timeseries(net:ModelState, dataset:Dataset, latent=False, seed=2553, save=False, use_conv=False, warp_imgs=True, include_initial=True, use_resNet=False, feature_size=128*128):
    """
    Plots the distribution of losses over time
    """
    tot_loss = np.zeros((net.model.time_steps_img * 7, len(dataset)))
    batch_size = 1024
    loader = dataset.create_batches(batch_size=batch_size, shuffle=False)
    foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256, 256), jitter_type=None,
                                                jitter_amount=0, device='cuda', warp_imgs=warp_imgs)
    counter = 0
    torch.manual_seed(seed)
    for batch, fixations in loader:
        recurrent_state_pred = None
        batch = batch.to('cuda')
        fixations = fixations.to('cuda')
        if use_resNet:
            seq = foveal_transform(batch.reshape(batch.shape[0], 3, 256, 256), fixations)
        else:
            seq = foveal_transform(batch.reshape(batch.shape[0], 1, 256, 256), fixations)
        seq = seq.to('cuda')
        if not net.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
        with torch.no_grad():
            for i in range(seq.shape[1]):
                image = seq[:, i]
                if use_conv:
                    image = image.reshape(image.shape[0], 128, 128)
                for t in range(net.model.time_steps_img):
                    if t >= net.model.time_steps_img - net.model.time_steps_cords:
                        pred, recurrent_state_pred = net.predict(image, fixations[:, i], recurrent_state_pred)
                    else:
                        pred, recurrent_state_pred = net.predict(image, torch.zeros_like(fixations[:, i]), recurrent_state_pred)
                    if i != 0 or include_initial:
                        if not use_resNet:
                            losses = torch.mean(torch.abs(pred[:, :image.shape[1]] + image), dim=list(range(1, len(pred.shape)))).cpu()
                            tot_loss[i * net.model.time_steps_img + t, counter:counter+batch.shape[0]] = losses
                        else:
                            tot_loss += torch.mean(torch.abs(pred + net.model.flatten(net.model.resNet(seq[:, i])))) * batch.shape[0]
        counter += batch.shape[0]
    mean_loss = np.mean(tot_loss, axis=1)
    std_loss = np.std(tot_loss, axis=1)
    conf_int_90 = scipy.stats.norm.interval(confidence=0.9, loc=mean_loss, scale=scipy.stats.sem(tot_loss, axis=1))
    conf_int_99 = scipy.stats.norm.interval(confidence=0.99, loc=mean_loss, scale=scipy.stats.sem(tot_loss, axis=1))
    x = range(net.model.time_steps_img * 7)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(x, mean_loss)
    ax.set_ylim((0, 0.4))
    for i in range(net.model.time_steps_img * 7):
        ax.scatter([i] * tot_loss.shape[1], tot_loss[i], alpha=0.1, color='lightblue', s=1)
    ax.fill_between(x, mean_loss-std_loss, mean_loss+std_loss, alpha=0.2, color='gray')
    ax.fill_between(x, mean_loss-2*std_loss, mean_loss+2*std_loss, alpha=0.2, color='gray')
    save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/loss_time_series', bbox_inches='tight')


def checkLossesInhibitory(feedback):
    print(feedback.shape)
    means = feedback.mean(dim=1).cpu().numpy()
    print(means.shape)
    print(np.percentile(means, 0))
    print(np.percentile(means, 0.5))
    print(np.percentile(means, 50))
    print(np.percentile(means, 99.5))
    print(np.percentile(means, 100))

def pred_rdm(model_feedback, ideal_feedback):
    cos = torch.nn.CosineSimilarity(dim=1)
    # cos = torch.nn.CosineSimilarity(dim=2)
    model_feedback = model_feedback[:, :, 0, :].cpu()
    ideal_feedback = ideal_feedback[:, :, 0, :].cpu()
    # model_feedback = model_feedback.repeat_interleave(7, dim=1)
    # ideal_feedback = ideal_feedback.repeat(1, 7, 1)
    # # print(feedback_pre_lesion[0, :, 0], feedback[0, :, 0])
    # similarity = cos(ideal_feedback, model_feedback)
    # print(similarity.shape)
    # similarity = torch.mean(similarity, dim=0)
    # print(similarity.shape)
    # similarity = similarity.reshape(7, 7)

    similarity = torch.zeros((7, 7))
    correlations = torch.zeros((2051, 7, 7))

    for ideal in range(7):
        for model in range(7):
            # similarity[ideal, model] = torch.mean(cos(model_feedback[:, model], ideal_feedback[:, ideal]), dim=0)
            corrs = torch.zeros((2051, 1))
            for i in range(2051):
                corrs[i] = torch.corrcoef(torch.stack((model_feedback[i, model, :], ideal_feedback[i, ideal, :])))[0, 1]
            similarity[ideal, model] = torch.nanmean(corrs.flatten())
            correlations[:, ideal, model] = corrs.flatten()

    sns.set_context('talk')
    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    sns.heatmap(similarity, cmap='plasma', ax=ax)
    ax.set_xlabel('lesioned model internal drive')
    ax.set_ylabel('ideal inhibition')
    save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/feedbackRdm', bbox_inches='tight')
    return similarity, correlations

def makeRDMs(similarity_full, similarity_lesion, corrs_full, corrs_lesioned):
    sns.set_context('talk')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    sns.heatmap(similarity_full[1:, 1:], cmap='plasma', ax=ax1, vmin=min(similarity_full[:, 1:].min(), similarity_lesion[:, 1:].min()), vmax=max(similarity_full[:, 1:].max(), similarity_lesion[:, 1:].max()))
    ax1.set_xlabel('full model internal drive')
    ax1.set_ylabel('ideal inhibition')
    ax1.set_xticklabels(['1', '2', '3', '4', '5', '6'])
    ax1.set_yticklabels(['1', '2', '3', '4', '5', '6'])

    sns.heatmap(similarity_lesion[1:, 1:], cmap='plasma', ax=ax2, vmin=min(similarity_full[:, 1:].min(), similarity_lesion[:, 1:].min()), vmax=max(similarity_full[:, 1:].max(), similarity_lesion[:, 1:].max()))
    ax2.set_xlabel('lesioned model internal drive')
    ax2.set_xticklabels(['1', '2', '3', '4', '5', '6'])
    ax2.set_yticklabels(['1', '2', '3', '4', '5', '6'])
    save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/feedbackRdm', bbox_inches='tight')

    modelRDM_pred = torch.zeros((7, 7))
    modelRDM_delayed = torch.zeros((7, 7))
    for i in range(6):
        modelRDM_pred[i, i] = 1
        modelRDM_delayed[i, i+1] = 1
    modelRDM_pred[6, 6] = 1
    corrs = torch.corrcoef(torch.stack((similarity_full.flatten(), similarity_lesion.flatten(), modelRDM_pred.flatten(), modelRDM_delayed.flatten())))
    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    # ax.bar(['ideal prediction,\nfull model', 'ideal prediction,\nlesioned model', 'previous crop,\nfull model', 'previous prediction,\nlesioned model'], [corrs[2, 0], corrs[2, 1], corrs[3, 0], corrs[3, 1]])
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.bar(['1', '2', '3', '4'], [corrs[2, 0], corrs[2, 1], corrs[3, 0], corrs[3, 1]])
    print(corrs[2, 0], corrs[2, 1], corrs[3, 0], corrs[3, 1])
    # save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/correlationRDMs', bbox_inches='tight')

    correlations = torch.zeros((2051, 4))
    for i in range(2051):
        corrs = torch.corrcoef(torch.stack((corrs_full[i].flatten(), corrs_lesioned[i].flatten(), modelRDM_pred.flatten(), modelRDM_delayed.flatten())))
        correlations[i] = torch.FloatTensor([corrs[2, 0], corrs[2, 1], corrs[3, 0], corrs[3, 1]])
    print(corrs.shape)
    print(correlations[:, 0].mean(), correlations[:, 1].mean(), correlations[:, 2].mean(), correlations[:, 3].mean())
    print(scipy.stats.ttest_ind(correlations[:, 1], correlations[:, 0], alternative='less'))
    print(scipy.stats.ttest_ind(correlations[:, 2], correlations[:, 3], alternative='less'))
    print(scipy.stats.ttest_1samp(correlations[:, 0], 0))
    print(scipy.stats.ttest_1samp(correlations[:, 1], 0))
    print(scipy.stats.ttest_1samp(correlations[:, 2], 0))
    print(scipy.stats.ttest_1samp(correlations[:, 3], 0))

    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    labels = np.ones((2051, 4))
    labels[:, 1] *= 2
    labels[:, 2] *= 3
    labels[:, 3] *= 4
    labels = labels.flatten()
    # ax.bar(['ideal prediction,\nfull model', 'ideal prediction,\nlesioned model', 'previous crop,\nfull model', 'previous prediction,\nlesioned model'], [corrs[2, 0], corrs[2, 1], corrs[3, 0], corrs[3, 1]])
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # data =  np.array([correlations[:, 0], correlations[:, 1], correlations[:, 2], correlations[:, 3]]).flatten()
    print(labels.shape, correlations.cpu().numpy().flatten().shape)
    sns.barplot({'labels': labels, 'data': correlations.cpu().numpy().flatten()}, x='labels', y='data', errorbar=('ci', 95), ax=ax, ci=99, edgecolor='black')
    # ax.bar(['1', '2', '3', '4'], [corrs[2, 0], corrs[2, 1], corrs[3, 0], corrs[3, 1]])
    print(corrs[2, 0], corrs[2, 1], corrs[3, 0], corrs[3, 1])
    save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/correlationRDMs', bbox_inches='tight')

    


def extract_rf_sizes(net:ModelState, use_conv=False):
    """
    extract receptive field sizes
    Was discontinued and is not used
    """
    if not use_conv:
        stimulus = torch.ones(1, 128* 128)
    else:
        stimulus = torch.ones(1, 128, 128)
    stimulus = stimulus.to('cuda')
    stimulus.requires_grad = True
    fixations = torch.zeros(1,2)
    fixations = fixations.to('cuda')
    recurrent_state = None
    shape = (4,4)
    _, _, recurrent_state = net.model(stimulus, fixations, recurrent_state)
    _, loss_terms, _ = net.model(stimulus, fixations, recurrent_state)
    # pred, _ = net.predict(stimulus, fixations, recurrent_state)
    for hidden_idx in [1, 2]:
        hidden_states = loss_terms[1][hidden_idx][0]
        imgs = []
        if not use_conv:
            for idx in range(0, 2048, 2048//16):
                gradient = torch.autograd.grad(hidden_states[idx], stimulus, retain_graph=True)
                pred_img = gradient[0][0].detach().reshape(128, 128).cpu().numpy()
                imgs.append(pred_img)
        else:
            for idx in range(0, 128, 128//16):
                gradient = torch.autograd.grad(hidden_states[0, 1][idx, idx], stimulus, retain_graph=True)
                pred_img = gradient[0][0].detach().cpu().numpy()
                imgs.append(pred_img)
        fig, _ = display(imgs, lims=None, shape=shape, figsize=(24,24), axes_visible=False, layout='tight')
        save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/receptive_field_layer{hidden_idx}', bbox_inches='tight')


def check_grid_cells(net:ModelState, dataset:Dataset, layer=[1, 2]):
    """
    Plots the activations of hidden units relative to global x and y both as raw data and without dataset biases
    """
    batch_size = 1024
    for layer_idx in layer:
        position_fixations = np.zeros((128, 128))
        if layer_idx == 0:
            activity_patterns = np.zeros((128, 128, 128*128+25))
        else:
            activity_patterns = np.zeros((128, 128, 2048))
        loader = dataset.create_batches(batch_size=batch_size, shuffle=True)
        for batch, fixations in loader:
            with torch.no_grad():
                activations = net.get_activations(batch, fixations.detach(), layer=layer_idx)
                fixations = fixations.numpy()
                fixations = (fixations + 1) * 64
                fixations = fixations.astype(int)
                if batch.shape[0] == batch_size:
                    for i in range(batch_size):
                        for time_step in range(1, 7):
                            activity_patterns[fixations[i, time_step, 0], fixations[i, time_step, 1]] += activations[i, time_step*6+5].cpu().numpy()
                            position_fixations[fixations[i, time_step, 0], fixations[i, time_step, 1]] += 1
        imgs = []
        if layer_idx == 0:
            for i in range(2048):
                imgs.append(activity_patterns[:, :, 128*128+99-i])
        else:
            for i in range(2048):
                imgs.append(activity_patterns[:, :, i])
        fig, _ = display(imgs, lims=(0, sorted([img.max() for img in imgs])[2000]), shape=(64,32), figsize=(64,32), axes_visible=False, layout='tight', cmap='hot')
        save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/spatial_cell_activation_layer{layer_idx}', bbox_inches='tight')

        
        activity_patterns_noBias = activity_patterns.copy()
        for x in range(128):
            for y in range(128):
                if position_fixations[x, y] != 0:
                    activity_patterns_noBias[x, y, :] /= position_fixations[x, y]
                else:
                    activity_patterns_noBias[x, y, :] *= 0
        for unit in range(activity_patterns_noBias.shape[2]):
            activity_patterns_noBias[:, :, unit] /= activity_patterns_noBias[:, :, unit].max()
        imgs_noBias = []
        for i in range(2048):
            imgs_noBias.append(activity_patterns_noBias[:, :, i])
        fig, _ = display(imgs_noBias, lims=(0, 1), shape=(64,32), figsize=(64,32), axes_visible=False, layout='tight', cmap='hot')
        save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/spatial_cell_activation_layer_noBias{layer_idx}', bbox_inches='tight')


def checkCellAvgActivity(net:ModelState, dataset:Dataset, layer=[1, 2]):
    """
    Test whether allocentric units are more or less active than avg unit
    """
    batch_size = 1024
    all_activations = None
    all_activations_allocentric = None
    loader = dataset.create_batches(batch_size=batch_size, shuffle=True)
    for batch, fixations in loader:
        activations = None
        activations_allocentric = None
        for layer_idx in layer:
            with torch.no_grad():
                if activations is None:
                    new_activations = net.get_activations(batch, fixations.detach(), layer=layer_idx, timestep=0).cpu()
                    activations = new_activations
                else:
                    new_activations = net.get_activations(batch, fixations.detach(), layer=layer_idx, timestep=0).cpu()
                    activations = torch.cat((activations, new_activations), dim=2)
                if layer_idx > 0:
                    if activations_allocentric is None:
                        activations_allocentric = new_activations[:, :, torch.unique(torch.cat((torch.from_numpy(net.model.lesion_map[(layer_idx-1)*2]), torch.from_numpy(net.model.lesion_map[(layer_idx-1)*2+1]))))]
                    else:
                        activations_allocentric = torch.cat((activations_allocentric, new_activations[:, :, torch.unique(torch.cat((torch.from_numpy(net.model.lesion_map[(layer_idx-1)*2]), torch.from_numpy(net.model.lesion_map[(layer_idx-1)*2+1]))))]), dim=2)
                if layer_idx > 0:
                    print(new_activations.shape, activations_allocentric.shape)
                else:
                    print(new_activations.shape)

        if all_activations is None:
            all_activations = activations
            all_activations_allocentric = activations_allocentric
        else:
            all_activations = torch.cat((all_activations, activations), dim=0)
            all_activations_allocentric = torch.cat((all_activations_allocentric, activations_allocentric), dim=0)
    print(all_activations.shape, all_activations_allocentric.shape)
    print(f"Mean, median and std all activations: {all_activations.mean()}, {all_activations.median()}, {all_activations.std()}")
    print(f"Mean, median and std allocentric unit activations: {all_activations_allocentric.mean()}, {all_activations_allocentric.median()}, {all_activations_allocentric.std()}")
    print(f"T-Test that mean of all units is less than mean of allo units: {scipy.stats.ttest_ind(all_activations.flatten(), all_activations_allocentric.flatten(), alternative='less')}")

def checkPredCells(net:ModelState, dataset:Dataset, layer=[1, 2]):
    """
    tests for prediction cells as done in Ali et al.
    Was not used for final thesis
    """
    batch_size = 1024
    all_activations = None
    for layer_idx in layer:
        all_activations_layer = None
        loader = dataset.create_batches(batch_size=batch_size, shuffle=True)
        for batch, fixations in loader:
            with torch.no_grad():
                activations = net.get_activations(batch, fixations.detach(), layer=layer_idx)
            if all_activations_layer is None:
                all_activations_layer = activations
            else:
                all_activations_layer = torch.cat((all_activations_layer, activations), dim=0)
        if all_activations is None:
            all_activations = all_activations_layer
        else:
            all_activations = torch.cat((all_activations, all_activations_layer), dim=2)
    all_activations_flat = all_activations.flatten(start_dim=0, end_dim=1)
    medians = torch.median(all_activations_flat, dim=0)[0]
    medians_abs_dev = torch.median(torch.abs(all_activations_flat - medians), dim=0)[0]
    stds = medians_abs_dev * 1.4826
    is_pred_unit = medians - (1.64 * stds) > 0

    return is_pred_unit.cpu().numpy(), np.sum(is_pred_unit.cpu().numpy())



def checkTimeScales(net:ModelState, dataset:Dataset):
    """
    check the time scales of the different layers
    """
    batch_size = 1024
    all_activations = None
    activation_series_layers = []
    for layer_idx in range(3):
        all_activations_layer = None
        loader = dataset.create_batches(batch_size=batch_size, shuffle=True)
        for batch, fixations in loader:
            with torch.no_grad():
                activations = net.get_preactivations(batch, fixations.detach(), layer=layer_idx)
            if all_activations_layer is None:
                all_activations_layer = activations
            else:
                all_activations_layer = torch.cat((all_activations_layer, activations), dim=0)
        if all_activations is None:
            all_activations = all_activations_layer
        else:
            all_activations = torch.cat((all_activations, all_activations_layer), dim=2)
        activation_series_layers.append(all_activations_layer)

    cos_sim = torch.nn.CosineSimilarity(dim=1)

    # check for differences in activity change time scales between layers
    for layer_idx in range(3):
        changesL1 = []
        changesL2 = []
        changesCos = []
        activation_changes = []
        for t in range(activation_series_layers[layer_idx].shape[1] - 1):
            changesL1.append(torch.norm(activation_series_layers[layer_idx][:, t+1, :] - activation_series_layers[layer_idx][:, t, :], p=1).cpu().numpy())
            changesL2.append(torch.norm(activation_series_layers[layer_idx][:, t+1, :] - activation_series_layers[layer_idx][:, t, :], p=2).cpu().numpy())
            changesCos.append(cos_sim(activation_series_layers[layer_idx][:, t+1, :],  activation_series_layers[layer_idx][:, t, :]).cpu().numpy())
            activation_changes.append(torch.mean(torch.abs(activation_series_layers[layer_idx][:, t+1, :] -  activation_series_layers[layer_idx][:, t, :])).cpu().numpy())
            # changesCos.append(torch.dot(activation_series_layers[layer_idx][:, t+1, :], activation_series_layers[layer_idx][:, t, :] / 
            #                             (torch.norm(activation_series_layers[layer_idx][:, t+1, :], p=2) *
            #                              torch.norm(activation_series_layers[layer_idx][:, t, :], p=2))).cpu().numpy())
        # print(f'Mean manhattan between timesteps for layer {layer_idx}: {np.mean(changesL1)}')
        print(f'Mean change per unit of layer {layer_idx}: {np.abs(np.mean(changesL1)) / (activation_series_layers[layer_idx].cpu().numpy().shape[2] * activation_series_layers[layer_idx].cpu().numpy().shape[0])}')
        print(f'Mean euclidean distance between timesteps for layer {layer_idx}: {np.abs(np.mean(changesL2) / activation_series_layers[layer_idx].cpu().numpy().shape[0])}')
        print(f'Mean activation change relative to mean activity for layer {layer_idx}: {np.abs(np.mean(activation_changes)) / np.abs(np.mean(activation_series_layers[layer_idx].cpu().numpy()))}')
        print(f'Mean cosine similarity between timesteps for layer {layer_idx}: {np.abs(np.mean(changesCos))} \n')
    



            


def extract_energy_usage(net:ModelState, dataset:Dataset):
    """
    Returns energy usage of network (L1 loss on all layers for given data set)
    """
    if net.model.use_conv:
        loss_fn='l1_allconv'
    else:
        loss_fn='l1_all'
    tot_loss = 0
    state = None
    batch_size = 1024
    loader = dataset.create_batches(batch_size=batch_size, shuffle=True)
    counter = 0
    for batch, fixations in loader:
        with torch.no_grad():
            loss, res, state = test_batch(net, batch, fixations, loss_fn, state)
        tot_loss += loss * batch.shape[0]
        counter += batch.shape[0] * net.model.time_steps_img
    tot_loss /= counter
    print(f"Average energy consumption per unit: {tot_loss}")


def test_knowledge_integration(net: ModelState, dataset: Dataset, device, latent=False, seed=2553, save=False):
    """
    Fixates one position multiple times and compares, if later fixation has a better prediction than the earlier. Works only for MNIST, was not continued later
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    batches, fixations = dataset.create_list_batches(batch_size=-1, sequence_length=10, shuffle=False)
    batches = batches[0].permute(1,0).reshape(-1, 1, 140, 56)
    losses = np.zeros(5)
    for current_pair_number in range(5):
        fixations = torch.ones((batches.shape[0], 10, 2))
        fixations[:, :, 0] *= -0.5
        fixations[:, 1, 0] += 1
        fixations[:, 3, 0] += 1
        fixations[:, 5, 0] += 1
        fixations[:, 7, 0] += 1
        fixations[:, 9, 0] += 1
        fixations[:, :, 1] *= (current_pair_number - 2) * 0.4

        foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(140, 56), jitter_type=None,
                                           jitter_amount=0, device=device)

        ex_seq = foveal_transform(batches, fixations)

        # Make fixations egocentric
        if not net.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
        fixations[:, :, 1] = fixations[:, :, 1] / 0.4

        h = net.model.init_state(fixations.shape[0])
        losses_even = []
        losses_odd = []
        recurrent_state = None
        for i in range(10):
            with torch.no_grad():
                h, l_a, recurrent_state = net.model(ex_seq[:, i], fixations[:, i, :], state=h, recurrent_state=recurrent_state)
                if i % 2 == 0:
                    losses_even.append(torch.mean(torch.abs(h)).cpu().numpy())
                else:
                    losses_odd.append(torch.mean(torch.abs(h)).cpu().numpy())
        for i in range(5):
            losses[i] += losses_even[i] + losses_odd[i]
    print(losses)


def test_activation_shift(net: ModelState, dataset: Dataset, device, latent=False, seed=2553, save=False):
    """
    Test whether activation gets rerouted from the periphery to the fovea, did not yield results and was not used for final thesis
    """
    if seed != None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    batches, fixations = dataset.create_list_batches(batch_size=-1, sequence_length=10, shuffle=False)
    batches = batches[0].permute(1,0).reshape(-1, 1, 140, 56)
    shifts = np.zeros((2, 54, 54))
    preds = np.zeros((4, 54, 54))
    fixations = torch.ones((batches.shape[0], 4, 2)) * 0.5
    fixations[:, 0, 0] += 1
    fixations[:, 1, 0] -= 1
    fixations[:, 2, 1] += 1
    fixations[:, 3, 1] -= 1
    fixations[:, :, 0] = fixations[:, :, 0] / 2 - 1
    fixations[:, :, 1] = fixations[:, :, 1] / 5 - 1
    
    for i in range(fixations.shape[1]):
        fixations[:, i] = fixations[:, i] - torch.ones((batches.shape[0], 2)) * 0.5


    # h = net.model.init_state(fixations.shape[0])
    # for i in range(4):
    #     net.zero_grad()
    #     # TODO make function in model that gives heat map which input units contributed to which output units
    #     # TODO Maybe also just for ones-matrix once for all 4 directions of fixation shifts? Not for actualy images? or if for images not summed over all images?
    #     state = torch.autograd.Variable(torch.ones_like(h), requires_grad=True)
    #     fixation = torch.autograd.Variable(fixations[:, i, :], requires_grad=True)
    #     pred = net.predict(state, fixation)
    #     # for x in range(54):
    #     #     for y in range(54):
    #     activity_shift = torch.autograd.grad(torch.sum(pred), state)[0][0].detach().numpy()
    #     shifts[i] = activity_shift
    #     preds[i] = pred.detach().reshape(-1,54,54)[0]
    stimulus_centers = [(0.1, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)]
    h = net.model.init_state(fixations.shape[0])
    net.zero_grad()
    # TODO make function in model that gives heat map which input units contributed to which output units
    # TODO Maybe also just for ones-matrix once for all 4 directions of fixation shifts? Not for actualy images? or if for images not summed over all images?
    x = np.linspace(0, 1, 54)
    # dim1 = multivariate_normal.pdf(x, mean=stimulus_centers[stimulus][0], cov=0.05)
    # dim2 = multivariate_normal.pdf(x, mean=stimulus_centers[stimulus][1], cov=0.05)
    # dim1 /= np.max(dim1)
    # dim2 /= np.max(dim2)
    # stimuli.append(np.atleast_2d(dim1).T @ np.atleast_2d(dim2))
    # stim = torch.from_numpy(np.atleast_2d(dim1).T @ np.atleast_2d(dim2)).flatten().repeat(h.shape[0], 1).float()
    stim = torch.ones((h.shape[0], 54*54))
    stim.requires_grad = True
    state = torch.ones_like(h)
    state.requires_grad = True
    fixation = torch.Tensor(fixations[:, 0, :])
    fixation.requires_grad = True
    out, _, recurrent_state = net.model(stim, fixation)
    # hidden_img = torch.autograd.grad(torch.sum(state), stim, retain_graph=True)[0][0].detach().reshape(54, 54).numpy()
    # hidden_fix = np.zeros((54, 54, 2))
    # for x in range(54):
    #     for y in range(54):
    #         hidden_fix[x, y] = torch.autograd.grad(state[0, x*54 + y], fixation, retain_graph=True)[0][0].detach().numpy()
    pred, _ = net.predict(stim, fixation, recurrent_state)
    # preds[i] = pred.detach().reshape(-1,54,54)[0]
    # pred_hidden = torch.autograd.grad(torch.sum(pred), state, retain_graph=True)[0][0].detach().reshape(54, 54).numpy()
    # hidden_img = torch.autograd.grad(torch.sum(state), stim, retain_graph=True)[0][0].detach().reshape(54, 54).numpy()
    pred_img = torch.autograd.grad(torch.sum(pred), stim, retain_graph=True)[0][0].detach().reshape(54, 54).numpy()
    pred_fix = np.zeros((54, 54, 2))
    for x in range(54):
        for y in range(54):
            pred_fix[x, y] = torch.autograd.grad(pred[0, x*54 + y], fixation, retain_graph=True)[0][0].detach()
    # shifts[0] = np.multiply(np.multiply(pred_hidden, hidden_img), hidden_fix[:, :, 0])
    # shifts[1] = np.multiply(np.multiply(pred_hidden, hidden_img), hidden_fix[:, :, 1])
    # fig, axes = display([pred_fix[:, :, 0], pred_fix[:, :, 1]], lims=None, shape=(2,1), figsize=(3,3), axes_visible=False, layout='tight')
    # fig, axes = display([shifts[0], shifts[1], pred_hidden, hidden_img, hidden_fix[:, :, 0], hidden_fix[:, :, 1]], lims=None, shape=(6,1), figsize=(3,3), axes_visible=False, layout='tight')
    fig, axes = display([pred_img], lims=None, shape=(1,1), figsize=(3,3), axes_visible=False, layout='tight')
    save_fig(fig, f'Results/Fig2/pred_img', bbox_inches='tight')
    fig, axes = display([np.multiply(pred_img, pred_fix[:, :, 0]), np.multiply(pred_img, pred_fix[:, :, 1])], lims=None, shape=(2,1), figsize=(3,3), axes_visible=False, layout='tight')
    save_fig(fig, f'Results/Fig2/pred_img_pred_fix', bbox_inches='tight')
    fig, axes = display([pred_fix[:, :, 0], pred_fix[:, :, 1]], lims=None, shape=(2,1), figsize=(3,3), axes_visible=False, layout='tight')
    save_fig(fig, f'Results/Fig2/pred_fix', bbox_inches='tight')

def plot_weight_distributions(net: ModelState):
    """
    Plot the distribution of weights split by bottom-up, lateral ad top-down
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    nbins=30
    for layer_idx in [1, 2]:
        layer = net.model.layers[layer_idx]
        bottom_up = layer.bottom_up.weight.detach().flatten().cpu().numpy()
        lateral = layer.lateral.weight.detach().flatten().cpu().numpy()
        top_down = layer.top_down.weight.detach().flatten().cpu().numpy()
        axs[layer_idx - 1, 0].hist(bottom_up, color='blue', bins=nbins)
        axs[layer_idx - 1, 0].set_title(f"bottom up layer {layer_idx}")
        axs[layer_idx - 1, 0].axvline(bottom_up.mean(), color='black', linestyle='dashed', linewidth=1)
        axs[layer_idx - 1, 1].hist(lateral, color='blue', bins=nbins)
        axs[layer_idx - 1, 1].set_title(f"lateral layer {layer_idx}")
        axs[layer_idx - 1, 1].axvline(lateral.mean(), color='black', linestyle='dashed', linewidth=1)
        axs[layer_idx - 1, 2].hist(top_down, color='blue', bins=nbins)
        axs[layer_idx - 1, 2].set_title(f"top down layer {layer_idx}")
        axs[layer_idx - 1, 2].axvline(top_down.mean(), color='black', linestyle='dashed', linewidth=1)
        print(f"Layer {layer_idx}, bottom up: {bottom_up.mean()}, lateral: {lateral.mean()}, top down: {top_down.mean()}")
    save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/histograms_weights', bbox_inches='tight')


def plot_histograms(losses):
    """
    Plot the histograms of model v baseline comparisons. Depending on how many losses are given either histograms for 1 model (1 row) or two models (2 rows) are plotted
    """
    keys = list(losses.keys())
    if len(keys) <= 12:
        fig, axs = plt.subplots(1, 8, figsize=(80, 10))
        axs[0].hist(losses[keys[0]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[0]].min(), losses[keys[2]].min()), max(losses[keys[0]].max(), losses[keys[2]].max())), bins=20)
        axs[0].hist(losses[keys[2]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[0]].min(), losses[keys[2]].min()), max(losses[keys[0]].max(), losses[keys[2]].max())), bins=20)
        axs[0].set_title('Comparison against average image')
        axs[0].legend()
        axs[1].hist(losses[keys[0]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[0]].min(), losses[keys[4]].min()), max(losses[keys[0]].max(), losses[keys[4]].max())), bins=20)
        axs[1].hist(losses[keys[4]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[0]].min(), losses[keys[4]].min()), max(losses[keys[0]].max(), losses[keys[4]].max())), bins=20)
        axs[1].set_title('Comparison against average color')
        axs[1].legend()
        axs[2].hist(losses[keys[10]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[10]].min(), losses[keys[6]].min()), max(losses[keys[10]].max(), losses[keys[6]].max())), bins=20)
        axs[2].hist(losses[keys[6]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[10]].min(), losses[keys[6]].min()), max(losses[keys[10]].max(), losses[keys[6]].max())), bins=20)
        axs[2].set_title('Comparison against previous fixation')
        axs[2].legend()
        axs[3].hist(losses[keys[1]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[1]].min(), losses[keys[3]].min()), max(losses[keys[1]].max(), losses[keys[3]].max())), bins=20)
        axs[3].hist(losses[keys[3]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[1]].min(), losses[keys[3]].min()), max(losses[keys[1]].max(), losses[keys[3]].max())), bins=20)
        axs[3].set_title('Comparison against Average image without initial')
        axs[3].legend()
        axs[4].hist(losses[keys[1]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[1]].min(), losses[keys[5]].min()), max(losses[keys[1]].max(), losses[keys[5]].max())), bins=20)
        axs[4].hist(losses[keys[5]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[1]].min(), losses[keys[5]].min()), max(losses[keys[1]].max(), losses[keys[5]].max())), bins=20)
        axs[4].set_title('Comparison against Average color without initial')
        axs[4].legend()
        axs[5].hist(losses[keys[7]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[7]].min(), losses[keys[8]].min()), max(losses[keys[7]].max(), losses[keys[8]].max())), bins=20)
        axs[5].hist(losses[keys[8]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[7]].min(), losses[keys[8]].min()), max(losses[keys[7]].max(), losses[keys[8]].max())), bins=20)
        axs[5].set_title('Comparison against previous fixation for fixation onset')
        axs[5].legend()
        axs[6].hist(losses[keys[0]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[0]].min(), losses[keys[9]].min()), max(losses[keys[0]].max(), losses[keys[9]].max())), bins=20)
        axs[6].hist(losses[keys[9]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[0]].min(), losses[keys[9]].min()), max(losses[keys[0]].max(), losses[keys[9]].max())), bins=20)
        axs[6].set_title('Comparison against model with shuffled fixation order')
        axs[6].legend()
        axs[7].hist(losses[keys[7]], label='model', alpha=0.5, color='blue', range=(min(losses[keys[7]].min(), losses[keys[11]].min()), max(losses[keys[7]].max(), losses[keys[11]].max())), bins=20)
        axs[7].hist(losses[keys[11]], label='baseline', alpha=0.5, color='green', range=(min(losses[keys[7]].min(), losses[keys[11]].min()), max(losses[keys[7]].max(), losses[keys[11]].max())), bins=20)
        axs[7].set_title('Comparison against model with shuffled fixation order just onset')
        axs[7].legend()
        save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/histograms_losses', bbox_inches='tight')
    else:
        plot_all = False
        if plot_all:
            fig, axs = plt.subplots(8, 2, figsize=(20, 80))
        else:
            fig, axs = plt.subplots(2, 4, figsize=(40, 20))
        nbins = 40
        name_model_1 = 'targeted lesion'
        name_model_2 = 'random lesion'
        font_size_title = 25
        legend_size = 25
        if plot_all:
            axs[0, 0].hist(losses[keys[0]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 0].hist(losses[keys[2]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 0].axvline(losses[keys[2]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[0, 0].axvline(losses[keys[0]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[0, 1].hist(losses[keys[11]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 1].hist(losses[keys[2]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 1].axvline(losses[keys[2]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[0, 1].axvline(losses[keys[11]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[0, 0].set_title('Average image', fontdict={'fontsize': font_size_title})
            axs[0, 0].legend()
            axs[0, 1].legend()
            axs[1, 0].hist(losses[keys[0]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 0].hist(losses[keys[4]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 1].hist(losses[keys[11]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 1].hist(losses[keys[4]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 0].set_title('Average color', fontdict={'fontsize': font_size_title})
            axs[1, 0].legend()
            axs[1, 1].legend()
            axs[2, 0].hist(losses[keys[10]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[10]].max(), losses[keys[6]].max(), losses[keys[14]].max())), bins=nbins)
            axs[2, 0].hist(losses[keys[6]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[10]].max(), losses[keys[6]].max(), losses[keys[14]].max())), bins=nbins)
            axs[2, 1].hist(losses[keys[14]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[10]].max(), losses[keys[6]].max(), losses[keys[14]].max())), bins=nbins)
            axs[2, 1].hist(losses[keys[6]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[10]].max(), losses[keys[6]].max(), losses[keys[14]].max())), bins=nbins)
            axs[2, 0].set_title('Previous fixation', fontdict={'fontsize': font_size_title})
            axs[2, 0].legend()
            axs[2, 1].legend()
            axs[3, 0].hist(losses[keys[1]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[1]].max(), losses[keys[3]].max(), losses[keys[12]].max())), bins=nbins)
            axs[3, 0].hist(losses[keys[3]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[1]].max(), losses[keys[3]].max(), losses[keys[12]].max())), bins=nbins)
            axs[3, 1].hist(losses[keys[12]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[1]].max(), losses[keys[3]].max(), losses[keys[12]].max())), bins=nbins)
            axs[3, 1].hist(losses[keys[3]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[1]].max(), losses[keys[3]].max(), losses[keys[12]].max())), bins=nbins)
            axs[3, 0].set_title('Average image without initial', fontdict={'fontsize': font_size_title})
            axs[3, 0].legend()
            axs[3, 1].legend()
            axs[4, 0].hist(losses[keys[1]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[1]].max(), losses[keys[5]].max(), losses[keys[12]].max())), bins=nbins)
            axs[4, 0].hist(losses[keys[5]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[1]].max(), losses[keys[5]].max(), losses[keys[12]].max())), bins=nbins)
            axs[4, 1].hist(losses[keys[12]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[1]].max(), losses[keys[5]].max(), losses[keys[12]].max())), bins=nbins)
            axs[4, 1].hist(losses[keys[5]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[1]].max(), losses[keys[5]].max(), losses[keys[12]].max())), bins=nbins)
            axs[4, 0].set_title('Average color without initial', fontdict={'fontsize': font_size_title})
            axs[4, 0].legend()
            axs[4, 1].legend()
            axs[5, 0].hist(losses[keys[7]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[5, 0].hist(losses[keys[8]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[5, 1].hist(losses[keys[13]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[5, 1].hist(losses[keys[8]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[5, 0].set_title('Previous fixation for fixation onset', fontdict={'fontsize': font_size_title})
            axs[5, 0].legend()
            axs[5, 1].legend()
            axs[6, 0].hist(losses[keys[0]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[9]].max(), losses[keys[15]].max(), losses[keys[11]].max())), bins=nbins)
            axs[6, 0].hist(losses[keys[9]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[9]].max(), losses[keys[15]].max(), losses[keys[11]].max())), bins=nbins)
            axs[6, 1].hist(losses[keys[11]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[9]].max(), losses[keys[15]].max(), losses[keys[11]].max())), bins=nbins)
            axs[6, 1].hist(losses[keys[15]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[9]].max(), losses[keys[15]].max(), losses[keys[11]].max())), bins=nbins)
            axs[6, 0].set_title('Model with shuffled fixation order', fontdict={'fontsize': font_size_title})
            axs[6, 0].legend()
            axs[6, 1].legend()
            axs[7, 0].hist(losses[keys[7]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[7, 0].hist(losses[keys[16]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[7, 1].hist(losses[keys[13]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[7, 1].hist(losses[keys[17]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[7, 0].set_title('Model with shuffled fixation order just onset', fontdict={'fontsize': font_size_title})
            axs[7, 0].legend()
            axs[7, 1].legend()
        else:
            axs[0, 0].hist(losses[keys[2]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 0].hist(losses[keys[0]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 0].axvline(losses[keys[2]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[0, 0].axvline(losses[keys[0]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[1, 0].hist(losses[keys[2]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 0].hist(losses[keys[11]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[2]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 0].axvline(losses[keys[2]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[1, 0].axvline(losses[keys[11]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[0, 0].set_title('Average image', fontdict={'fontsize': font_size_title})
            axs[0, 0].legend(prop={'size': legend_size})
            axs[1, 0].legend(prop={'size': legend_size})
            axs[0, 1].hist(losses[keys[0]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 1].hist(losses[keys[4]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[0, 1].axvline(losses[keys[4]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[0, 1].axvline(losses[keys[0]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[1, 1].hist(losses[keys[11]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 1].hist(losses[keys[4]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[0]].max(), losses[keys[4]].max(), losses[keys[11]].max())), bins=nbins)
            axs[1, 1].axvline(losses[keys[4]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[1, 1].axvline(losses[keys[11]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[0, 1].set_title('Average color', fontdict={'fontsize': font_size_title})
            axs[0, 1].legend(prop={'size': legend_size})
            axs[1, 1].legend(prop={'size': legend_size})
            axs[0, 2].hist(losses[keys[7]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[0, 2].hist(losses[keys[8]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[0, 2].axvline(losses[keys[8]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[0, 2].axvline(losses[keys[7]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[1, 2].hist(losses[keys[13]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[1, 2].hist(losses[keys[8]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[8]].max(), losses[keys[13]].max())), bins=nbins)
            axs[1, 2].axvline(losses[keys[8]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[1, 2].axvline(losses[keys[13]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[0, 2].set_title('Previous fixation for fixation onset', fontdict={'fontsize': font_size_title})
            axs[0, 2].legend(prop={'size': legend_size})
            axs[1, 2].legend(prop={'size': legend_size})
            axs[0, 3].hist(losses[keys[7]], label=f'model {name_model_1}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[0, 3].hist(losses[keys[16]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[0, 3].axvline(losses[keys[16]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[0, 3].axvline(losses[keys[7]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[1, 3].hist(losses[keys[13]], label=f'model {name_model_2}', alpha=0.5, color='blue', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[1, 3].hist(losses[keys[17]], label='baseline', alpha=0.5, color='green', range=(0.0, max(losses[keys[7]].max(), losses[keys[16]].max(), losses[keys[13]].max(), losses[keys[17]].max())), bins=nbins)
            axs[1, 3].axvline(losses[keys[17]].mean(), color='green', linestyle='dashed', linewidth=1)
            axs[1, 3].axvline(losses[keys[13]].mean(), color='blue', linestyle='dashed', linewidth=1)
            axs[0, 3].set_title('Model with shuffled fixation order just onset', fontdict={'fontsize': font_size_title})
            axs[0, 3].legend(prop={'size': legend_size})
            axs[1, 3].legend(prop={'size': legend_size})
        save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/histograms_losses', bbox_inches='tight')



def ttests(losses):
    """
    Compute t-tests to compare model to baselines
    """
    keys = list(losses.keys())
    print(f't-test model vs shuffled: {scipy.stats.ttest_rel(losses[keys[0]], losses[keys[9]], alternative="less")}')
    print(f't-test model vs avg image: {scipy.stats.ttest_ind(losses[keys[0]], losses[keys[2]], alternative="less")}')
    print(f't-test model vs avg color: {scipy.stats.ttest_ind(losses[keys[0]], losses[keys[4]], alternative="less")}')
    print(f't-test model vs avg image without initial: {scipy.stats.ttest_ind(losses[keys[1]], losses[keys[3]], alternative="less")}')
    print(f't-test model vs avg color without initial: {scipy.stats.ttest_ind(losses[keys[1]], losses[keys[5]], alternative="less")}')
    print(f't-test model vs previous fixation: {scipy.stats.ttest_ind(losses[keys[10]], losses[keys[6]], alternative="less")}')
    print(f't-test model vs previous fixation just fixation onset: {scipy.stats.ttest_ind(losses[keys[7]], losses[keys[8]], alternative="less")}')

