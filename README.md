# Predictive remapping and allocentric coding as consequences of energy efficiency in recurrent neural network models of active vision

### Description
Codebase for the paper "Predictive remapping and allocentric coding as consequences of energy efficiency in recurrent neural network models of active vision"

### Dependencies

```pip install -r requirements.txt```

Or look in requirements.txt - be sure to use Python >=3.7

### Training

- The model can be trained by calling "src/train_models.py"

- Make sure to adjust the dataset path in "src/train_models.py"

- The path for the saved model can be adjusted in the function "save" in "src/ModelState.py"

### Evaluation

- To select the model to be loaded, adjust the model name and path in the function "load" in "src/ModelState.py"

- Calling the script "src/analyseModels.py" extracts the data for Fig2B/C and Fig3C/D and stores it in svg files. Additionally, the plots for Fig2D, Fig3A/B/E/F/G/ are created.

- Calling the script "src/plotResults.py" creates the plots for Figure 2B/C and Figure 3D. It additionally performs and prints the necessary t-tests for Figures 2B and 3D. Make sure to adjust the path for the svg file at the top of the script of necessary.

- Calling the script "src/plotWeights.py" creates Figure 3C. Make sure to adjust the path for the svg file at the top of the script of necessary.

- All created plots and svg files are stored in "src/Results/Fig2_mscoco/".

### Explanation of remaining files

#### RNN.py
The file containing all model architectures and model logic

#### train_models.py
The file setting all training and model hyperparameters and calling the respective functions for training and testing from train.py

#### train.py
The file with the logic for training and testing the model

#### plot.py
The file containing all plotting and analysis functions that are called in fig2_network_performance.py

#### ClosedFormDecoding.py
- regressionCoordinates: Trains and tests a decoding model for coordinates, returns indices of rped units and decoder weights
- regressionTime: Trains and tests a decoding model for time

#### ModelState.py
A Wrapper for all models

#### Dataset.py
Wrapper class for datasets

#### H5dataset.py
The file loading the msCOCO - Deepgaze - dataset. Running the file, the exemplory image used in Fig 1 is plotted. 

#### mnist.py
Not used anymore

#### FovelaTransform.py
A torch layer performing the foveal transform. The mechanic is not used anymore. If warp_imgs is est to False, the layer simply performs quadratic crops around the given fixation coordinates and returns the list of crops

#### functions.py
Loss functions and other helpful functions.

### Files not used in the final version of the paper but still referenced in the codebase:

#### DecodingModel.py
A decoding model as a torch model that can be rained using gradient descent

#### train_decoding_model.py
File to train the torch decoding model

#### test_decoding_model.py
file to test the torch decoding model

#### GridCellCoding.py
Creates a torch layer that converts global x-y into grid cell activations.

#### ResNet.py
Pre-trained ResNet18 for extracting visual features from the image to test predictive coding in higher visual features than pixels.
