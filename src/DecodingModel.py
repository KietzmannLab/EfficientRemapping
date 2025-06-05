import torch
import numpy as np
from ModelState import ModelState
import functions
import plot
from matplotlib import pyplot as plt


class DecodingModel(torch.nn.Module):
    """
    A linear decoding model trained with gradient descent. Was not used for the thesis as the closed form solution could be run as well
    """

    def __init__(self, net, layer, device, hidden_size, time_step, use_pred_units=False, test_set=None):
        super(DecodingModel, self).__init__()
        self.net = net
        self.layer = layer
        self.device = device
        self.use_pred_units = use_pred_units
        if self.use_pred_units:
            self.is_pred_unit, hidden_size = plot.checkPredCells(net, test_set, layer=[1, 2])
            # hidden_size = 120
            # hidden_size = 33
            # hidden_size = 7
            hidden_size = 16
        self.transform = torch.nn.Linear(hidden_size, 2)
        self.time_step = time_step
        if self.use_pred_units:
            self.time_step = None
        


        
    
    def forward(self, batch, fixations):
        with torch.no_grad():
            if not self.use_pred_units:
                activations = self.net.get_activations(batch, fixations.detach(), self.layer)
            else:
                activations = torch.cat((self.net.get_activations(batch, fixations.detach(), 1), self.net.get_activations(batch, fixations.detach(), 2)), dim=2)
                activations = activations[:, :, self.is_pred_unit]
        if self.time_step is not None:
            results = torch.zeros_like(fixations)
            for i in range(activations.shape[1] // self.net.model.time_steps_img):
                pred = self.transform(activations[:, i * self.net.model.time_steps_img + self.time_step])
                results[:, i] = pred
        else:
            results = torch.zeros(fixations.shape[0], activations.shape[1], fixations.shape[2])
            for i in range(activations.shape[1]):
                pred = self.transform(activations[:, i])
                results[:, i] = pred
        return results
    
class DecodingModelState(ModelState):

    def __init__ (self, optimizer, lr, title, device, net, layer, hidden_size, time_step=0, test_next_fixation=False, use_pred_units=False, test_set=None):
        ModelState.__init__(self,

                            DecodingModel(net, layer, device, hidden_size, time_step, use_pred_units, test_set).to(device),
                            optimizer,
                            lr,
                            title,
                            {
                                "train loss": np.zeros(0),
                                "test loss": np.zeros(0),
                                "h": np.zeros(0),
                                "Wl1": np.zeros(0),
                                "Wl2": np.zeros(0)
                            },
                            device,
                            is_decoding_model=True)
        self.loss_fn = torch.nn.MSELoss()
        self.test_next_fixation = test_next_fixation
        self.mean_fixation = torch.zeros((1,2))
        
    def run(self, batch, fixations, loss_fn, state):
        results = self.model(batch, fixations)
        if self.model.time_step is not None:
            if not self.test_next_fixation:
                loss = self.loss_fn(results, fixations)
                sum_squares = self.loss_fn(results, self.mean_fixation.repeat(7, 1).unsqueeze(0).repeat(results.shape[0], 1, 1))
            else:                
                loss = self.loss_fn(results[:, :-1], fixations[:, 1:])
                sum_squares = self.loss_fn(results[:, :-1], self.mean_fixation.repeat(7, 1).unsqueeze(0).repeat(results.shape[0], 1, 1)[:, 1:])
        else:
            if not self.test_next_fixation:
                fixations = torch.repeat_interleave(fixations, self.model.net.model.time_steps_img, dim=1)
                loss = self.loss_fn(results, fixations)
                sum_squares = self.loss_fn(results, self.mean_fixation.repeat(42, 1).unsqueeze(0).repeat(results.shape[0], 1, 1))
            else:
                fixations = torch.repeat_interleave(fixations[:, 1:], self.model.net.model.time_steps_img, dim=1)
                loss = self.loss_fn(results[:, :-self.model.net.model.time_steps_img], fixations)
                sum_squares = self.loss_fn(results[:, :-1], self.mean_fixation.repeat(42, 1).unsqueeze(0).repeat(results.shape[0], 1, 1)[:, 1:])
        r_squared = 1 - (loss / sum_squares)
        return loss, r_squared, None


    def step(self, loss, scaler=None):
        loss.backward()
        self.optimizer.step()
    

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_mean_fix(self, mean_fix):
        self.mean_fixation = mean_fix


    def save(self, epoch=0):
        filepath = "./EmergentPredictiveCoding/models/" + self.title + str(epoch) + ".pth"

        if self.model.use_pred_units:
            torch.save({
                "epochs": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "results": self.results,
                'pred_units': self.model.is_pred_unit
            }, filepath)
        else:
            torch.save({
                "epochs": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "results": self.results
            }, filepath)

    def load(self, idx=None, path=None, twolayers=True):
        if not self.is_decoding_model:
            if (idx is None):
                filepath = "./models/" + self.title +".pth"
            else:
                # filepath = "./models/" + self.title +"_" + str(idx) + ".pth"
                # if twolayers:
                #     filepath = "./models/" + self.title +"_" + str(idx) + "_fc_1layer_2916_timesteps_1_1_ordered_1init_largerLR_1000.pth"
                # else:
                    # filepath = "./models/" + self.title +"_" + str(idx) + "_fc_0layer_lateral_2916_timesteps_1_1_ordered_1init_largerLR_1000.pth"
                if self.model.use_conv:
                    filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_conv_lateral_2layer_2048_timesteps_6_3_lr1e4_clipping_multistep_250.pth"
                else:
                    filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr15e5_clipping_1500.pth"
                    # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_lstm_2layer_2048_timesteps_6_3_1e4_scheduler_450.pth"
                    # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_scheduler_1500.pth"
                    # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_all_lateral_2layer_2048_timesteps_6_3_lr6e5_scheduler_850.pth"
                    # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_50_6_grid_cells_500.pth"
                    # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr8e5_scheduler_dropout_1000.pth"
                    # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_scheduler_changed_optim_350.pth"
                    # filepath = "./models/" + self.title +"_" + str(idx) + "lateral.pth"
            if path is not None:
                filepath = path
        else:
            # filepath = "./models/" + self.title +"_" + str(idx) + ".pth"
            # if twolayers:
            #     filepath = "./models/" + self.title +"_" + str(idx) + "_fc_1layer_2916_timesteps_1_1_ordered_1init_largerLR_1000.pth"
            # else:
            #     filepath = "./models/" + self.title +"_" + str(idx) + "_fc_0layer_lateral_2916_timesteps_1_1_ordered_1init_largerLR_1000.pth"
            filepath = "./EmergentPredictiveCoding/models/" + self.title + "25.pth"
            # filepath = "./models/" + self.title +"_" + str(idx) + "lateral.pth"

        state = torch.load(filepath, map_location=torch.device(self.device))
        self.epochs = state['epochs']
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.results = state['results']
        if self.model.use_pred_units:
            self.model.is_pred_unit = state['pred_units']