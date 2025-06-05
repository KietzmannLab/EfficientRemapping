# we will read the efficient remapping RNN loss results and plot the performance


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import plot
import scipy

# ==============================================
# Data preparation for plots
# ==============================================

# read the data
losses_df = pd.read_csv('EmergentPredictiveCoding/Results/Fig2_mscoco/losses.csv')

# make the data long format 
losses_long = pd.melt(losses_df, id_vars=['Unnamed: 0'], value_vars=losses_df.columns[1:], var_name='model', value_name='loss')
# rename the index column

losses_long = losses_long.rename(columns={'Unnamed: 0': 'im'})


# unpack this into meaningful columns that show the model, the lesion, and the iterations (first number is the fixation, second number is the RNN time step)
losses_long[['model', 'fixation', 'rnn_iter']] = losses_long['model'].str.split(',', expand=True)

# remove the parentheses
losses_long['model'] = losses_long['model'].str.replace('(', '')
losses_long['rnn_iter'] = losses_long['rnn_iter'].str.replace(')', '')
# make the model and rnn_iter columns integers
losses_long['fixation'] = losses_long['fixation'].str.replace("'", '').astype(int)
losses_long['rnn_iter'] = losses_long['rnn_iter'].str.replace("'", '').astype(int)
# print(losses_long)

# remove unnecessary quotes
losses_long['model'] = losses_long['model'].str.replace("'", '')
losses_long['model'] = losses_long['model'].str.strip()

# make the model column a category
losses_long['model'] = losses_long['model'].astype('category')
print(losses_long.keys)


# make a dictionary to map the model names to shorter names
model_map = {
    'Avg color': 'average train\nluminance', # average color of the test set
    'Avg image': 'average train crop', # average image of the test set
    'Avg local crop': 'location specific average\ntrain crop',
    'Model lesion': 'lesioned:\nallocentric units', # lesioning of the allocentric units
    'Model random Lesion': 'lesioned:\nrandom units', # lesioning of random units
    'Model': 'full model', # full model with all the jazz
    'Model small crops': 'full model\nsmall crops',
    'Model no efference': 'full model\nno efference',
    'Prev Fixation': 'previous fixation\ncrop', # previous fixation crop
    'Shuffled Fixation': 'shuffled fixation\nsequence', # shuffled fixation crop (with the same image?)
    'Shuffled fixation Lesioned': 'lesioned:\nallocentric units\n+ shuffled fixation', # please elaborate on this
    'Shuffled fixation random Lesion': 'lesioned:\nrandom units\n+ shuffled fixation' # lesioning of random units and shuffled fixation
}



# make the model column a category
losses_long['model'] = losses_long['model'].astype('category')
# rename the model column
for key, value in model_map.items():
    losses_long['model'] = losses_long['model'].str.replace(key, value)
# set hue order for plotting
hue_order = ['full model', 'average train\nluminance', 'average train crop', 'previous fixation\ncrop', 
             'shuffled fixation\nsequence', 'full model', 'lesioned:\nallocentric coding units', 'lesioned:\nrandom units',
             'lesioned:\nallocentric units\n+ shuffled fixation', 'lesioned:\nrandom units\n+ shuffled fixation']
# print(hue_order == 'full model')

# make full model emerald green
full_color = '#27ae60'


# print(np.unique(losses_long['model']))

# can you make table that shows the number of zeros in the loss for each model, fixation, and rnn_iter
zero_loss_table = losses_long.groupby(['model', 'fixation', 'rnn_iter'])['loss'].apply(lambda x: np.sum(x == 0)).reset_index()
zero_loss_table = zero_loss_table.rename(columns={'loss': 'zero_loss_count'})
zero_loss_table.to_csv('zero_loss_table.csv', index=False)

# for the previous fixation, only want to use time step 0 but never fixation 0


# howmany are masked
# add acolum that shows the rnn_iter over the full fixation sequence
losses_long['rnn_timestep_scene'] = losses_long['fixation'] * 6 + losses_long['rnn_iter']




# count the number of zeros in the loss per model
losses_long['zero_loss'] = losses_long['loss'] == 0
# print(losses_long.groupby('model')['zero_loss'].sum())

# remove the zeros from the data
losses_long = losses_long[~losses_long['zero_loss']]


# introduce a newline in the model names if longer than 3 words

# # plot histograms of the losses for each model 
sns.set_context('talk')

# make model categories


# make a new column for lesioned models
losses_long['lesioned'] = losses_long['model'].str.contains('lesioned')
# make this a category with label lesioned and intact
losses_long['lesioned'] = losses_long['lesioned'].replace({True: 'lesioned', False: 'intact'}).astype('category')

# # subplot for non-lesioned models

# separate the data into lesioned and non-lesioned models
non_lesioned_models = losses_long[losses_long['lesioned'] == 'intact']
lesioned_models = losses_long[losses_long['lesioned'] == 'lesioned']

# add the full model to the lesioned models
lesioned_models = pd.concat([non_lesioned_models[non_lesioned_models['model'] == 'full model'], lesioned_models])
hue_order_non_lesioned = hue_order[:len(non_lesioned_models['model'].unique())]
hue_order_lesioned = hue_order[len(non_lesioned_models['model'].unique()):]

pallette_greys_non_lesioned = sns.color_palette('Greys', len(non_lesioned_models['model'].unique()) +2)[2:]
pallette_greys_lesioned = sns.color_palette('Greys', len(lesioned_models['model'].unique()) +2)[2:]
# set the color of the full model to emerald green (it occurs two times in the data)
pallette_greys_non_lesioned[0] = full_color
pallette_greys_lesioned[0] = full_color
pallette_greys_lesioned[1] = 'tomato'


# only use first rnn iteration
non_lesioned_models = non_lesioned_models[non_lesioned_models['rnn_iter'] == 0]
lesioned_models = lesioned_models[lesioned_models['rnn_iter'] == 0]

for model in np.unique(lesioned_models['model']):
    print(model, (lesioned_models[lesioned_models['model'] == model]['loss']).shape)
for model in np.unique(non_lesioned_models['model']):
    print(model, (non_lesioned_models[non_lesioned_models['model'] == model]['loss']).shape)

# ==============================================
# Make significance tests for comparison of models and controls
# ==============================================

print("\n\nindependent t-tests full model vs baselines (alternative that full model loss is less)")
for test_pair in [('full model', 'average train\nluminance'), ('full model', 'average train crop'), ('full model', 'previous fixation\ncrop'), ('full model', 'shuffled fixation\nsequence'), ('full model', 'full model small crops'), ('full model', 'full model no efference'), ('full model', 'location specific average\ntrain crop')]:
    # print((non_lesioned_models[non_lesioned_models['model'] == test_pair[0]])['loss'].shape, (non_lesioned_models[non_lesioned_models['model'] == test_pair[1]])['loss'].shape)
    print(test_pair[1], scipy.stats.ttest_ind(non_lesioned_models[non_lesioned_models['model'] == test_pair[0]]['loss'], non_lesioned_models[non_lesioned_models['model'] == test_pair[1]]['loss'], alternative='less'))

print("\n\nindependent t-tests full model small crops vs baselines (alternative that full model loss is less)")
for test_pair in [('full model small crops', 'average train\nluminance'), ('full model small crops', 'average train crop'), ('full model small crops', 'previous fixation\ncrop'), ('full model small crops', 'shuffled fixation\nsequence'), ('full model small crops', 'full model no efference'), ('full model small crops', 'location specific average\ntrain crop')]:
    # print((non_lesioned_models[non_lesioned_models['model'] == test_pair[0]])['loss'].shape, (non_lesioned_models[non_lesioned_models['model'] == test_pair[1]])['loss'].shape)
    print(test_pair[1], scipy.stats.ttest_ind(non_lesioned_models[non_lesioned_models['model'] == test_pair[0]]['loss'], non_lesioned_models[non_lesioned_models['model'] == test_pair[1]]['loss'], alternative='less'))

print("\n\nindependent t-tests full model no efference vs baselines (alternative that full model loss is less)")
for test_pair in [('full model no efference', 'average train\nluminance'), ('full model no efference', 'average train crop'), ('full model no efference', 'previous fixation\ncrop'), ('full model no efference', 'shuffled fixation\nsequence'), ('full model no efference', 'full model small crops'), ('full model no efference', 'location specific average\ntrain crop')]:
    # print((non_lesioned_models[non_lesioned_models['model'] == test_pair[0]])['loss'].shape, (non_lesioned_models[non_lesioned_models['model'] == test_pair[1]])['loss'].shape)
    print(test_pair[1], scipy.stats.ttest_ind(non_lesioned_models[non_lesioned_models['model'] == test_pair[0]]['loss'], non_lesioned_models[non_lesioned_models['model'] == test_pair[1]]['loss'], alternative='less'))

print("\n\nindependent t-tests full model vs both lesioned and lesioned allocentric vs both shuffled lesioned ones (alternative that full model/ not shuffled loss is less)")
for test_pair in [('full model', 'lesioned:\nallocentric units'), ('full model', 'lesioned:\nrandom units'), ('lesioned:\nallocentric units', 'lesioned:\nallocentric units\n+ shuffled fixation'), ('lesioned:\nallocentric units', 'lesioned:\nrandom units\n+ shuffled fixation')]:
    # print((non_lesioned_models[non_lesioned_models['model'] == test_pair[0]])['loss'].mean(), (non_lesioned_models[non_lesioned_models['model'] == test_pair[1]])['loss'].mean())
    print(test_pair[1], scipy.stats.ttest_ind(lesioned_models[lesioned_models['model'] == test_pair[0]]['loss'], lesioned_models[lesioned_models['model'] == test_pair[1]]['loss'], alternative='less'))

print("\n\nindependent t-tests random lesioned model vs allocentric lesioned and shuffled randm lesioned model vs shuffled allocentric model (alternative that random lesioned model loss is less)")
for test_pair in [('lesioned:\nrandom units', 'lesioned:\nallocentric units'), ('lesioned:\nrandom units\n+ shuffled fixation', 'lesioned:\nallocentric units\n+ shuffled fixation')]:
    print((lesioned_models[lesioned_models['model'] == test_pair[0]])['loss'].mean(), (lesioned_models[lesioned_models['model'] == test_pair[1]])['loss'].mean())
    print(test_pair[1], scipy.stats.ttest_ind(lesioned_models[lesioned_models['model'] == test_pair[0]]['loss'], lesioned_models[lesioned_models['model'] == test_pair[1]]['loss'], alternative='less'))

print("\n\nindependent t-tests allocentric lesioned model vs baselines (alternative that allocentric lesioned loss is less)")
for test_pair in [('lesioned:\nallocentric units', 'average train\nluminance'), ('lesioned:\nallocentric units', 'average train crop'), ('lesioned:\nallocentric units', 'previous fixation\ncrop'), ('lesioned:\nallocentric units', 'shuffled fixation\nsequence'), ('lesioned:\nallocentric units', 'full model small crops'), ('lesioned:\nallocentric units', 'full model no efference')]:
    # print((non_lesioned_models[non_lesioned_models['model'] == test_pair[0]])['loss'].mean(), (non_lesioned_models[non_lesioned_models['model'] == test_pair[1]])['loss'].mean())
    print(test_pair[1], scipy.stats.ttest_ind(lesioned_models[lesioned_models['model'] == test_pair[0]]['loss'], non_lesioned_models[non_lesioned_models['model'] == test_pair[1]]['loss'], alternative='less'))

print("\n\nindependent t-tests random lesioned model vs baselines (alternative that random lesioned loss is less)")
for test_pair in [('lesioned:\nrandom units', 'average train\nluminance'), ('lesioned:\nrandom units', 'average train crop'), ('lesioned:\nrandom units', 'previous fixation\ncrop'), ('lesioned:\nrandom units', 'shuffled fixation\nsequence')]:
    # print((non_lesioned_models[non_lesioned_models['model'] == test_pair[0]])['loss'].mean(), (non_lesioned_models[non_lesioned_models['model'] == test_pair[1]])['loss'].mean())
    print(test_pair[1], scipy.stats.ttest_ind(lesioned_models[lesioned_models['model'] == test_pair[0]]['loss'], non_lesioned_models[non_lesioned_models['model'] == test_pair[1]]['loss'], alternative='less'))


# ==============================================
# Make analysis for loss per foxation distance and create plot
# ==============================================
fixations = pd.read_csv('EmergentPredictiveCoding/Results/Fig2_mscoco/fixations.csv')

fixations = pd.melt(fixations, id_vars=['Unnamed: 0'], value_vars=fixations.columns[1:], var_name='time', value_name='fixation')
fixations[['coordinate', 'fixation_nr']] = fixations['time'].str.split(',', expand=True)
fixations['coordinate'] = fixations['coordinate'].str.replace('(', '')
fixations['fixation_nr'] = fixations['fixation_nr'].str.replace(')', '')
fixations['fixation_nr'] = fixations['fixation_nr'].str.replace("'", '').astype(int)
fixations['coordinate'] = fixations['coordinate'].str.replace("'", '')
fixations['fixation_nr'] = fixations['fixation_nr'].astype('category')

fixations_x = fixations[fixations['coordinate'] == 'x']
fixations_y = fixations[fixations['coordinate'] == 'y']

fixations_array = np.stack([np.stack((fixations_x[fixations['fixation_nr'] == i]['fixation'], fixations_y[fixations['fixation_nr'] == i]['fixation'])) for i in range(7)])
losses = non_lesioned_models[non_lesioned_models['model'] == 'full model']
loss_arrays = np.stack([losses[losses['fixation'] == i]['loss'] for i in range(7)])

losses_distance = np.zeros((6, 3))
number_losses = np.zeros((6, 3))
stds_losses = np.zeros((6, 3))

dists = None
for current_idx in range(1, 7):
    for second_idx in range(current_idx):
        vectors = fixations_array[current_idx] - fixations_array[second_idx]
        if dists is None:
            dists = np.sqrt(np.multiply(vectors[0], vectors[0]) + np.multiply(vectors[1], vectors[1]))
        else:
            dists = np.append(dists, np.sqrt(np.multiply(vectors[0], vectors[0]) + np.multiply(vectors[1], vectors[1])))
percentiles = np.percentile(dists, [100/3, 200/3])
print(percentiles)


for current_idx in range(1, 7):
    for second_idx in range(current_idx):
        vectors = fixations_array[current_idx] - fixations_array[second_idx]
        dists = np.sqrt(np.multiply(vectors[0], vectors[0]) + np.multiply(vectors[1], vectors[1]))
        small_dists = dists < percentiles[0]
        medium_dists = np.logical_and(dists >= percentiles[0], dists < percentiles[1])
        long_dists = dists >= percentiles[1]
        losses_distance[current_idx - second_idx - 1, 0] += np.sum(loss_arrays[current_idx, small_dists])
        number_losses[current_idx - second_idx - 1, 0] += np.sum(small_dists)
        stds_losses[current_idx - second_idx - 1, 0] = np.std(loss_arrays[current_idx, small_dists])
        losses_distance[current_idx - second_idx - 1, 1] += np.sum(loss_arrays[current_idx, medium_dists])
        number_losses[current_idx - second_idx - 1, 1] += np.sum(medium_dists)
        stds_losses[current_idx - second_idx - 1, 1] = np.std(loss_arrays[current_idx, medium_dists])
        losses_distance[current_idx - second_idx - 1, 2] += np.sum(loss_arrays[current_idx, long_dists])
        number_losses[current_idx - second_idx - 1, 2] += np.sum(long_dists)
        stds_losses[current_idx - second_idx - 1, 2] = np.std(loss_arrays[current_idx, long_dists])
losses_distance = np.divide(losses_distance, number_losses)
print(np.sum(number_losses[:, 0]), np.sum(number_losses[:, 1]), np.sum(number_losses[:, 2]))
print(losses_distance)
print(number_losses)
print(stds_losses)
print(f"Mean per distance: {np.mean(losses_distance, axis=0)}")
print(f"Mean per distance last two: {np.mean(losses_distance[:2], axis=0)}")
print(f"Mean per distance 2-3 away: {np.mean(losses_distance[1:3], axis=0)}")
mean_losses = np.mean(loss_arrays[1:])
std_losses = np.std(loss_arrays[1:])
loss_matrix = (losses_distance[:3] - mean_losses) / std_losses
print(loss_matrix)

fig = plt.figure(figsize=(3, 2.5))
ax = plt.gca()
sns.heatmap(loss_matrix, cmap='seismic', ax=ax, vmin=min(loss_matrix.min(), -1*loss_matrix.max()), vmax=max(loss_matrix.max(), -1*loss_matrix.min()))
ax.set_ylabel('time since fixation')
ax.set_xlabel('distance to fixation')
ax.set_yticklabels(['1', '2', '3'])
ax.invert_yaxis()
plot.save_fig(fig, f'EmergentPredictiveCoding/Results/Fig2_mscoco/SaccadeEffectMatrix', bbox_inches='tight')


# ==============================================
# Make boxplots for lesioned and non-lesioned models compared to controls
# ==============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [0.6, 0.4]})

sns.barplot(data=non_lesioned_models, x='model', y='loss', 
            palette=pallette_greys_non_lesioned,
            ax=ax1, ci=99, width=0.3, edgecolor='black',
            estimator=np.mean, errwidth=9)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_title('energy efficiency')
ax1.set_xlabel('')
ax1.set_ylabel('energy efficiency\n[loss]')

sns.barplot(data=lesioned_models, x='model', y='loss', 
            palette=pallette_greys_lesioned,
            ax=ax2, ci=99, width=0.3, edgecolor='black', errwidth=9,)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_xlabel('')
ax2.set_ylabel('energy efficiency\n[loss]')
ax2.set_ylabel('')


# decorate each bar with a dotted line full model loss
full_model_loss = non_lesioned_models[non_lesioned_models['model'] == 'full model']['loss'].mean()
for ax in [ax1, ax2]:
    ax.axhline(full_model_loss, linestyle='--', linewidth=3, label='full model\nloss', color='darkgrey', zorder=0)
ax2.legend(frameon=False, loc='upper left')
ax1.legend(frameon=False, loc='upper left')
# set the y limits to the same
ax1.set_ylim(0, 0.3)
ax2.set_ylim(0, 0.3)
plt.tight_layout()
sns.despine()
plot.save_fig(fig, 'EmergentPredictiveCoding/Results/Fig2_mscoco/barplot_lesion')

