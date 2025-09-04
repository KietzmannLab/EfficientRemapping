from ModelState import ModelState
from Dataset import Dataset
from sklearn.linear_model import LinearRegression
import torch
import numpy as np
import pandas as pd
import plot


def getFeatures(net:ModelState, dataset:Dataset, layer_idx, timestep=None, shuffle=False, index=None):
    batch_size = 1024
    all_activations_layer = None
    all_fixations = None
    loader = dataset.create_batches(batch_size=batch_size, shuffle=shuffle)
    if not isinstance(layer_idx, list):
         layer_idx = [layer_idx]
    for batch, fixations in loader:
        if index is not None:
            torch.manual_seed(2553)
            iterator = iter(dataset.create_batches(batch_size=1, shuffle=shuffle))
            for i in range(index):
                batch, fixations = next(iterator)
            # fig, axes = plot.display(batch, cmap='gray', lims=(0, 1), shape=(1,1), figsize=(6,6), axes_visible=False, layout='tight')
            # plot.save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/'+"test_decoding_input", bbox_inches='tight')
            batch = batch.reshape(1, 256, 256)
            fixations = fixations.reshape(1, 7, 2)
        with torch.no_grad():
            activations = None
            for layer in layer_idx:
                if activations is None:
                    activations = net.get_activations(batch, fixations.detach(), layer=layer, timestep=timestep).cpu()
                else:
                    activations = torch.cat((activations, net.get_activations(batch, fixations.detach(), layer=layer, timestep=timestep).cpu()), dim=2)
        if all_activations_layer is None:
            all_activations_layer = activations
        else:
            all_activations_layer = torch.cat((all_activations_layer, activations), dim=0)
        if all_fixations is None:
            all_fixations = fixations
        else:
            all_fixations = torch.cat((all_fixations, fixations), dim=0)
        if index is not None:
             break
    all_activations_layer = all_activations_layer.flatten(start_dim=0, end_dim=1).cpu().numpy()
    if timestep is None:
        all_fixations = torch.repeat_interleave(all_fixations.flatten(start_dim=0, end_dim=1), 6, 0).cpu().numpy()
    else:
        all_fixations = all_fixations.flatten(start_dim=0, end_dim=1).cpu().numpy()
    # print(f"return getFeatures: {all_activations_layer.shape, all_fixations.shape}")
    return all_activations_layer, all_fixations


def regressionCoordinates(net:ModelState, train_set:Dataset, test_set:Dataset, layer=[1, 2], mode='prev_relative', timestep=None):
    """
    mode: string, either 'prev_relative', 'next_relative' or 'global'
    """
    pred_units_all = None
    reg_weights_all = None
    # for layer_idx in layer:
    print("Getting training features")
    X_train, Y_train = getFeatures(net, train_set, layer_idx=layer, timestep=timestep)
    # print(f"Before relative: {np.isnan(X_train).sum(), np.isnan(Y_train).sum()}")
    if 'relative' in mode:
        Y_train = changeToRelativeCoordinates(Y_train, prev='prev' in mode, timestep=timestep)
        # if 'prev' in mode:
        #         Y_train = Y_train[:-6]
        #         X_train = X_train[6:]
    # print(f"After relative: {np.isnan(X_train).sum(), np.isnan(Y_train).sum()}")

    norm_factors = np.mean(X_train, axis=0)
    norm_factors += np.ones_like(norm_factors) * 0.000001
    X_train = X_train / norm_factors
    print("Fitting regression")
    reg = LinearRegression(n_jobs=-1).fit(X_train, Y_train, 2)
    # print("regression fitted")
    # print(f"Training score layer {layer_idx}: {reg.score(X_train, Y_train)}")
    print(f"Training score: {reg.score(X_train, Y_train)}")
    torch.manual_seed(2553)
    print("Getting test features")
    X_test, Y_test = getFeatures(net, test_set, layer_idx=layer, timestep=timestep)
    if 'relative' in mode:
        Y_test = changeToRelativeCoordinates(Y_test, prev='prev' in mode, timestep=timestep)
    X_test = X_test / norm_factors
    # print(f"Test score layer {layer_idx}: {reg.score(X_test, Y_test)}")
    test_score = reg.score(X_test, Y_test)
    print(f"Test score: {test_score}")

    pred_Y = reg.predict(X_test)
    target_mean = Y_test.mean(axis=0)
    s_res = np.sum(np.square(Y_test - pred_Y), axis=0)
    s_tot = np.sum(np.square(Y_test - target_mean), axis=0)
    r_squared = np.ones((1, 2)) - np.divide(s_res, s_tot)
    print(f"R squared: {r_squared}")
    # F_PATH = 'EmergentPredictiveCoding/Results/Fig2_mscoco/'
    # df = pd.DataFrame(np.concatenate((pred_Y, Y_test), axis=1), columns=['x_pred', 'y_pred', 'x_target', 'y_target'])
    # df.to_csv(F_PATH+"decoder_preds.csv")

    # torch.manual_seed(2553)
    # example_activations, example_fix = getFeatures(net, test_set, layer_idx=layer, timestep=timestep, index=1842)
    # example_activations = example_activations / norm_factors
    # example_pred = reg.predict(example_activations)
    # F_PATH = 'EmergentPredictiveCoding/Results/Fig2_mscoco/'
    # df = pd.DataFrame(np.concatenate((example_pred, example_fix), axis=1), columns=['x_pred', 'y_pred', 'x_target', 'y_target'])
    # df.to_csv(F_PATH+"example_preds.csv")

    reg_weights = reg.coef_
    pred_units = np.argsort(np.abs(reg_weights), axis=1)
    if pred_units_all is None:
            pred_units_all = pred_units
    else:
            pred_units_all = np.concatenate((pred_units_all, pred_units), axis=0)
    if reg_weights_all is None:
            reg_weights_all = np.sort(np.abs(reg_weights), axis=1)[:, ::-1]
            # reg_weights_all = reg_weights
    else:
            reg_weights_all = np.concatenate((reg_weights_all, np.sort(np.abs(reg_weights), axis=1)[:, ::-1]), axis=0)
            # reg_weights_all = np.concatenate((reg_weights_all, reg_weights), axis=0)
    # pred_units_all_2 = pred_units_all.copy()
    # pred_units_all = pred_units_all[pred_units_all < 2048]
    # pred_units_all_2 = pred_units_all_2[pred_units_all_2 >= 2048]
    # pred_units_all_2 = pred_units_all_2 - 2048
    # pred_units_all = np.concatenate((pred_units_all, pred_units_all_2), axis=0)
    # print(pred_units_all.shape, pred_units_all.max())
    return pred_units_all, reg_weights_all, test_score
    



def regressionTime(net:ModelState, train_set:Dataset, test_set:Dataset, layer=[1, 2]):
        for layer_idx in layer:
            X_train, _ = getFeatures(net, train_set, layer_idx=layer_idx)
            Y_train = torch.repeat_interleave(torch.arange(0, 6).reshape(6, 1), 7, dim=0)
            Y_train = Y_train.repeat(X_train.shape[0] // 42, 1).cpu().numpy()
            reg = LinearRegression().fit(X_train, Y_train)
            print("regression fitted")
            print(f"Training score layer {layer_idx}: {reg.score(X_train, Y_train)}")
            X_test, _ = getFeatures(net, test_set, layer_idx=layer_idx)
            Y_test = torch.repeat_interleave(torch.arange(0, 6).reshape(6, 1), 7, dim=0)
            Y_test = Y_test.repeat(X_test.shape[0] // 42, 1).cpu().numpy()
            print(f"Test score layer {layer_idx}: {reg.score(X_test, Y_test)}")

def changeToRelativeCoordinates(coordinates, prev=True, timestep=None):
    # print(f"Before change to relative: {coordinates.shape}")
    relative_coordinates = np.zeros_like(coordinates)
    if prev:
        for time_step in range(coordinates.shape[0]):
            if timestep is None:
                if int(time_step / 6) % 7 != 0:
                    relative_coordinates[time_step] = coordinates[time_step] - coordinates[time_step - 6]
            else:
                if time_step % 7 != 0:
                    relative_coordinates[time_step] = coordinates[time_step] - coordinates[time_step - 1]
    else:
         for time_step in range(coordinates.shape[0]):
            if timestep is None:
                if int(time_step / 6) % 7 != 6:
                    relative_coordinates[time_step] = coordinates[time_step + 6] - coordinates[time_step]
            else:
                if time_step % 7 != 6:
                    relative_coordinates[time_step] = coordinates[time_step + 1] - coordinates[time_step]
    # print(f"After change to relative: {relative_coordinates.shape}")
    return relative_coordinates
    