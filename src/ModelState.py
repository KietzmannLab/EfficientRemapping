import torch

class ModelState:
    """A class to encapsulate a neural network model with a number of attributes associated with it.

    Serves as a place to store associated attributes to a model, such as the optimizer or training metadata.
    """
    def __init__(self,
                 model,
                 optimizer,
                 lr:float,
                 title:str,
                 results,
                 device:str,
                 is_decoding_model=False):
        self.is_decoding_model = is_decoding_model
        self.model = model
        if is_decoding_model:
            self.optimizer = optimizer(self.model.transform.parameters(), lr=lr)
        else:
            if self.model.use_conv:
                self.optimizer = optimizer(self.model.parameters(), lr=lr)
            else:
                lr *= self.model.hidden_size
                input_weights = ['layer1.lateral.weight', 'layer2.bottom_up.weight']
                special_params = list(filter(lambda param: param[0] in input_weights, self.model.named_parameters()))
                base_params = list(filter(lambda param: param[0] not in input_weights, self.model.named_parameters()))
                # parameters = list(filter(lambda param: True, self.model.named_parameters()))
                self.optimizer = optimizer([
                    {'params': [param[1] for param in base_params], 'lr': lr/self.model.hidden_size},
                    {'params': [param[1] for param in special_params], 'lr': lr/self.model.input_size}
                    ])

        self.title = title
        self.epochs = 0
        self.results = results
        self.device = device
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=range(200, 1500, 200), gamma=0.75)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=range(50, 250, 50), gamma=0.75)

    def save(self, epoch=0):
        filepath = "./EmergentPredictiveCoding/models/" + self.title + str(epoch) + ".pth"

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
                    if not self.model.hybrid:
                        filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_conv_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_250.pth"
                    else:
                        if "all" in self.title:
                            filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_hybrid_2layer_2048_timesteps_6_3_lr5e5_partial_ReLU_250.pth"
                        else:
                            filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_hybrid_2layer_2048_timesteps_6_3_lr1e4_partial_ReLU_250.pth"
                        # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_hybrid_2layer_2048_timesteps_6_3_lr5e5_ReLU_Decay_250.pth"
                        # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_hybrid_2layer_2048_timesteps_6_3_lr1e4_ReLU_250.pth"
                else:
                    if self.model.input_size == 85*85:
                        filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_nonCords_new_moreRL_smallCrops_1500.pth"
                    elif not self.model.use_fixation:
                        filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_nonCords_new_moreRL_noCords_1500.pth"
                    else:
                        # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr5e4_ReLU_nonCords_1500.pth"
                        # filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_nonCords_new_lessDecay_1500.pth"
                        filepath = "./EmergentPredictiveCoding/models/" + self.title +"_" + str(idx) + "_fc_lateral_2layer_2048_timesteps_6_3_lr1e4_ReLU_nonCords_new_moreRL_1500.pth"
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

    def lr_schedule(self):
        # return
        self.scheduler.step()
