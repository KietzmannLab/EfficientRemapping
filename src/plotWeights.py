# plot the decoding of allocentric coordinates on the nework units
#

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plot

weights = pd.read_csv('Results/Fig2_mscoco/decoder_weights.csv')


weights.columns = ['units', 'x', 'y']


weights['layer'] = np.where(weights['units'] < 2048, 1, 2)

lesioned_units = {
    1: [363, 262, 1319, 578, 208, 266, 1840, 262],
    2: [1900, 787, 488, 839, 107, 820, 981, 1900, 3, 1510, 403, 2033, 251, 1345]}

# add a lesioned column
weights['lesioned'] = False
for layer, units in lesioned_units.items():
    if layer == 2:
        units = [unit + 2048 for unit in units]
    weights.loc[(weights['units'].isin(units)) & (weights['layer'] == layer), 'lesioned'] = True

print(weights['x'].shape, weights['y'].shape)
weights_long = np.concatenate((weights['x'], weights['y']))
# weights_long = weights['y']
print(weights_long.shape)
mean=weights_long.mean()
std = weights_long.std()
significant_weights = np.where(np.abs(weights_long) < mean + 1.645 * std, 0, 1)
print(mean, std, significant_weights.sum())
