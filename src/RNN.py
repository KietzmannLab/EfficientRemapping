import torch
import numpy as np
from ModelState import ModelState
import functions
from FovealTransform import FovealTransform
from GridCellCoding import GridCellCoding
from ResNet import ResNet

class RNN(torch.nn.Module):
    """
    Recurrent Neural Network class containing parameters of the network
    and computes the forward pass.
    Returns prediction of the input, a llist of potential loss terms and a list of hidden states
    """

    def __init__(self, input_size: int, hidden_size: int, activation_func,
                 prevbatch=False, device=None, use_fixation=True, use_lateral=True, use_conv=False, use_lstm=False, use_resNet=False, time_steps_img=1, time_steps_cords=1, twolayers=True, dropout_rate=0, disentangled_loss=False, useReservoir=False):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_func = activation_func
        self.use_fixation = use_fixation
        self.use_grid_coding = False
        self.use_recurrence_integration_layer = True
        self.use_conv = use_conv
        self.use_resNet = use_resNet
        self.hybrid = False
        self.lesion = False
        self.lesion_map = None
        self.relay_image = False
        self.hybrid_point = 3
        self.time_steps_img = time_steps_img
        self.time_steps_cords = time_steps_cords
        self.use_lstm = use_lstm
        self.dropout_rate = dropout_rate
        self.disentangled_loss = disentangled_loss
        self.useReservoir = useReservoir
        if self.use_grid_coding:
            self.number_frequencies = 6
            self.number_cells_frequency = 50
            self.grid_cell_layer = GridCellCoding(number_frequencies=self.number_frequencies, number_cells_frequency=self.number_cells_frequency, device=device)
        if self.use_fixation:
            if self.use_grid_coding:
                self.layer1 = RNN_block(input_size + self.number_frequencies * self.number_cells_frequency, input_size + self.number_frequencies * self.number_cells_frequency, output_size=input_size + self.number_frequencies * self.number_cells_frequency, l_connection=False, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func)
                self.layer2 = RNN_block(input_size + self.number_frequencies * self.number_cells_frequency, hidden_size, output_size=input_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
                self.layer3 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
                self.layers = [self.layer1, self.layer2, self.layer3]
            else:
                if self.use_conv:
                    if self.hybrid:
                        self.layer1 = RNN_block_conv(1, 1, 1, output_channel=1, l_connection=False, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func)
                        self.layer2 = RNN_block_conv(1, 32, 3, output_channel=1, l_connection=False, activation_func=activation_func, excitatory_b=False, pool_input=True)
                        self.layer3 = RNN_block_conv(32, 32, 3, output_channel=1, l_connection=False, activation_func=activation_func, excitatory_b=False, pool_input=True)
                        self.layer4 = RNN_block(input_size * 2 + 2, hidden_size, output_size = input_size*2, l_connection=True, activation_func=activation_func, excitatory_b=False, disregard_xy_ReLU=True)
                        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
                    else:
                        self.layer1 = RNN_block_conv(1, 1, 1, output_channel=3, l_connection=False, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func)
                        self.layer2 = RNN_block_conv(3, 32, 7, output_channel=1, l_connection=True, activation_func=activation_func, excitatory_b=True, pool_input=True)
                        self.layer3 = RNN_block_conv(32, 32, 7, l_connection=True, activation_func=activation_func, excitatory_b=True, pool_input=True)
                        self.layers = [self.layer1, self.layer2, self.layer3]
                elif use_resNet:
                    self.layer1 = RNN_block(input_size + 2, input_size + 2, output_size=input_size + 2, l_connection=False, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func)
                    self.layer2 = RNN_block(input_size + 2, hidden_size, output_size=input_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
                    self.layer3 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
                    self.layer4 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
                    self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
                elif use_lstm:
                    self.layer1 = RNN_block(input_size + 2, input_size + 2, output_size=input_size + 2, l_connection=False, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func)
                    self.layer2 = LSTM_block(input_size + 2, hidden_size, output_size=input_size, l_connection=True, excitatory_b=False, activation_func=activation_func)
                    self.layer3 = LSTM_block(hidden_size, hidden_size, excitatory_b=False, l_connection=False, activation_func=activation_func)
                    # self.layer4 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
                    self.layers = [self.layer1, self.layer2, self.layer3]
                elif self.useReservoir:
                    self.layer1 = RNN_block(input_size + 2, input_size + 2, output_size=input_size + 2, l_connection=True, t_connection=False, identity_b=True, activation_func=activation_func, disentangled_loss=self.disentangled_loss)
                    self.layers = [self.layer1]
                else:
                    # if twolayers:
                    self.layer1 = RNN_block(input_size + 2, input_size + 2, output_size=input_size + 2, l_connection=False, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func, disentangled_loss=self.disentangled_loss)
                    self.layer2 = RNN_block(input_size + 2, hidden_size, output_size=input_size, l_connection=True, activation_func=activation_func, excitatory_b=False, disentangled_loss=self.disentangled_loss)
                    self.layer3 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False, disentangled_loss=self.disentangled_loss)
                    # self.layer4 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
                    self.layers = [self.layer1, self.layer2, self.layer3]
                    # else:
                    #     self.layer1 = RNN_block(input_size + 2, input_size + 2, output_size=input_size + 2, l_connection=True, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func)
                    #     self.layers = [self.layer1]
        else:
            self.layer1 = RNN_block(input_size, input_size, output_size=input_size, l_connection=False, t_connection=False, identity_b=True, use_activation_func=False, activation_func=activation_func, disentangled_loss=self.disentangled_loss)
            self.layer2 = RNN_block(input_size, hidden_size, output_size=input_size, l_connection=True, activation_func=activation_func, excitatory_b=False, disentangled_loss=self.disentangled_loss)
            self.layer3 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False, disentangled_loss=self.disentangled_loss)
            # self.layer4 = RNN_block(hidden_size, hidden_size, l_connection=True, activation_func=activation_func, excitatory_b=False)
            self.layers = [self.layer1, self.layer2, self.layer3]
        self.prevbatch = prevbatch
        self.device = device

    def forward(self, x, fixation, state=None, recurrent_state=None):
        h = self.init_state(x.shape[0])
        if self.use_conv:
            h = h.reshape(h.shape[0], 128, 128)
        h = h.to(self.device)
        x = x.to(self.device)
        hidden_states = []
        next_top_down_states = []

        if self.use_fixation:
            if not self.use_conv:
                if self.use_grid_coding:
                    grid_code = self.grid_cell_layer(fixation)
                    x = torch.cat((x, grid_code), 1)
                else:
                    x = torch.cat((x, fixation), 1)
            elif not self.hybrid:
                # print (x.shape, fixation[:,0].shape, fixation[:, 0].repeat(128 * 128).reshape(128, 128, 1024).permute(2, 0, 1))
                fixation_x = fixation[:, 0].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                fixation_y = fixation[:, 1].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                x = torch.stack([x, fixation_x, fixation_y], 1)
            else:
                x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        if recurrent_state is None:
            recurrent_states = [None] * len(self.layers)
            top_down_states = [None] * len(self.layers)
            for i, layer in enumerate(self.layers):
                if i + 1 < len(self.layers):
                    t_state = top_down_states[i+1]
                else:
                    t_state = None
                if i == 0:
                    if self.use_fixation:
                        if not self.use_conv:
                            if self.use_grid_coding:
                                grid_code = self.grid_cell_layer(fixation)
                                h = torch.cat((h, grid_code), 1)
                            else:
                                h = torch.cat((h, fixation), 1)
                            if self.disentangled_loss:
                                top_down, y, _ = layer(h, recurrent_states[i], t_state)
                            else:
                                y, top_down, _ = layer(h, recurrent_states[i], t_state)
                            if self.relay_image:
                                y = h
                        elif not self.hybrid:
                            fixation_x = fixation[:, 0].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                            fixation_y = fixation[:, 1].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                            h = torch.stack([h, fixation_x, fixation_y], 1)
                            y, top_down, _ = layer(h, recurrent_states[i], t_state)
                        else:
                            h = h.reshape(h.shape[0], 1, h.shape[1], h.shape[2])
                            y, top_down, _ = layer(h, recurrent_states[i], t_state)
                    else:
                        y, top_down, _ = layer(h, recurrent_states[i], t_state)
                elif self.use_conv and self.hybrid and i == self.hybrid_point:
                    shape = y.shape
                    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2] * y.shape[3])
                    y, top_down, _ = layer(torch.cat((y, fixation), 1), recurrent_states[i], t_state)
                    top_down = top_down.reshape(shape)
                elif self.use_lstm:
                    top_down, y, rec_state = layer(y, recurrent_states[i], t_state)
                else:
                    if self.disentangled_loss:
                        top_down, y, _ = layer(y, recurrent_states[i], t_state)
                    else:
                        y, top_down, _ = layer(y, recurrent_states[i], t_state)
                if self.lesion and i != 0:
                # print(x[:, self.lesion_map[(i-1)*2]].shape)
                    y[:, self.lesion_map[(i-1)*2]] = 0
                    y[:, self.lesion_map[(i-1)*2+1]] = 0
                if not self.use_lstm or i==0:
                    hidden_states.append(y)
                else:
                    hidden_states.append((y, rec_state))
                next_top_down_states.append(top_down)
            recurrent_state = [hidden_states, next_top_down_states]
        loss_states = []
        hidden_states = []
        next_top_down_states = []
        recurrent_states = recurrent_state[0]
        top_down_states = recurrent_state[1]
        for i, layer in enumerate(self.layers):
            if i + 1 < len(self.layers):
                t_state = top_down_states[i+1]
            else:
                t_state = None
            if self.use_conv and self.hybrid and i == self.hybrid_point:
                shape = x.shape
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
                x, top_down, pre = layer(torch.cat((x, fixation), 1), recurrent_states[i], t_state)
                top_down = top_down.reshape(shape)
            else:
                if not self.use_lstm or i==0:
                    if self.disentangled_loss:
                        top_down, z, loss_terms = layer(x, recurrent_states[i], t_state)
                    else:
                        z, top_down, pre = layer(x, recurrent_states[i], t_state)
                elif self.use_lstm:
                    top_down, z, rec_state = layer(x, recurrent_states[i], t_state)
                else:
                    top_down, z, loss_terms = layer(x, recurrent_states[i], t_state)
                if not self.relay_image or not i == 0:
                    x = z
            if self.lesion and i != 0:
                # print(x[:, self.lesion_map[(i-1)*2]].shape)
                x[:, self.lesion_map[(i-1)*2]] = 0
                x[:, self.lesion_map[(i-1)*2+1]] = 0
                # x = torch.zeros_like(x)
                # top_down = torch.zeros_like(top_down)
            if self.relay_image:
                hidden_states.append(z)
            elif self.use_lstm and not i==0:
                hidden_states.append((x, rec_state))
            elif not self.disentangled_loss:
                hidden_states.append(x)
                loss_states.append(pre)
            else:
                hidden_states.append(x)
                loss_states.append(loss_terms)
            next_top_down_states.append(top_down)
            if i == 0:
                out = top_down
                if self.use_fixation and not self.hybrid:
                    if self.relay_image:
                        pre_act = z[:, :-2]
                    else:
                        pre_act = pre[:, :-2]
                else:
                    pre_act = pre
        # Top-down input to the input layer, [pre-activation of the input layer, pre_activation of all layers], [hidden states and top_down outputs of all layers]
        return out, [pre_act, loss_states, None], [hidden_states, next_top_down_states]
            

    def state_dict(self, *args, **kwargs):
        state_dict1 = super().state_dict(*args, **kwargs)
        if self.use_grid_coding:
            state_dict1.update({'grid_offsets': self.grid_cell_layer.offsets})
        return state_dict1

    def load_state_dict(self, state_dict, *args, **kwargs):
        if self.use_grid_coding:
            self.grid_cell_layer.offsets = state_dict['grid_offsets']
            del state_dict['grid_offsets']
        super().load_state_dict(state_dict, *args, **kwargs)
        return


    def init_state(self, batch_size):
        return torch.zeros((batch_size, self.input_size))
    
    def setLesionMap(self, lesion_map=None, random=False):
        self.lesion = True
        if not random:
            lesion_map = lesion_map[:, -11:]
            print(lesion_map.shape)
            self.lesion_map = [lesion_map.copy()[0], lesion_map.copy()[1], lesion_map.copy()[0], lesion_map.copy()[1]]
            for idx, elem in enumerate(self.lesion_map):
                if idx > 1:
                    elem = elem[elem >= 2048]
                    self.lesion_map[idx] = elem - 2048
                else:
                    self.lesion_map[idx] = elem[elem < 2048]
            for elem in self.lesion_map:
                print(elem)
        else:
            self.lesion_map = torch.randint(low=0, high=2048, size=(4, 5))
        # self.lesion_map = lesion_map
    

class ResNetRNN(RNN):
    """
    A wrapper calss around the RNN to allow for other networks (e.g. ResNet) to be used on the input as a feature extractor before being fed into the RNN.
    This idea was discarded, the wrapper class does not change the input
    """

    def __init__(self, input_size: int, hidden_size: int, activation_func,
                 prevbatch=False, device=None, use_fixation=True, use_lateral=True, use_conv=False, use_lstm=False, use_resNet=False, time_steps_img=1, time_steps_cords=1, twolayer=True, dropout=0, disentangled_loss=False, useReservoir=False):
        super(ResNetRNN, self).__init__(input_size, hidden_size, activation_func, prevbatch, device, use_fixation, use_lateral, use_conv, use_lstm, use_resNet, time_steps_img, time_steps_cords, twolayer, dropout, disentangled_loss, useReservoir)
        self.twolayer=twolayer
        if self.use_resNet:
            self.resNet = ResNet()
            for p in self.resNet.parameters():
                p.requires_grad = False
            self.flatten = torch.nn.Flatten()
    
    def forward(self, x, fixation, state=None, recurrent_state=None):
        if self.use_resNet:
            x = x.to(self.device)
            with torch.no_grad():
                x = self.resNet(x)
            x = self.flatten(x)
        out, loss_terms, recurrent_terms = super().forward(x, fixation, state, recurrent_state)
        return out, loss_terms, recurrent_terms
    

class RNN_block(torch.nn.Module):
    """
    A RNN block that gets bottom up, has a lateral connection and outputs a top-down connection.
    Blocks can be stacked on top of each other (allows for top-down input as well)
    """


    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 b_connection=True,
                 l_connection=True,
                 t_connection=True,
                 excitatory_b=False,
                 identity_b = False,
                 use_activation_func=True,
                 activation_func=None,
                 dropout=0,
                 disentangled_loss=False,
                 disregard_xy_ReLU=False):
        super(RNN_block, self).__init__()
        self.b_connection = b_connection
        self.l_connection = l_connection
        self.t_connection = t_connection
        self.use_activation_func = use_activation_func
        self.activation_func = activation_func
        self.dropout = dropout
        self.disentangled_loss = disentangled_loss
        self.disregard_xy_ReLU = disregard_xy_ReLU
        if dropout != 0:
            self.dropout_layer = torch.nn.Dropout(p=dropout)
        if output_size is None:
            self.output_size = input_size
        else:
            self.output_size = output_size
        self.input_size = input_size
        if b_connection and not identity_b:
            self.bottom_up = torch.nn.Linear(input_size, hidden_size)
            # torch.nn.init.normal_(self.bottom_up.weight, mean=0.0, std=1.0/input_size)
        if l_connection:
            self.lateral = torch.nn.Linear(hidden_size, hidden_size)
            # torch.nn.init.normal_(self.lateral.weight, mean=0.0, std=2.0/hidden_size)
        if t_connection:
            self.top_down = torch.nn.Linear(hidden_size, self.output_size)
            # torch.nn.init.normal_(self.top_down.weight, mean=0.0, std=2.0/math.sqrt(hidden_size))
        self.identity_b = identity_b
        self.excitatory_b = excitatory_b
        if excitatory_b:
            class WeightClipper(object):
                def __init__(self, frequency=5):
                    self.frequency = frequency
                def __call__(self, module):
                    if hasattr(module, 'weight'):
                        w = module.weight.data
                        w = w.clamp(0,1)
            clipper = WeightClipper()
            self.bottom_up.apply(clipper)
    
    def forward(self, x, recurrent_input=None, top_down_input=None):
        if self.b_connection and not self.identity_b:
            pre = self.bottom_up(x)
            bottom = pre.clone()
        else:
            pre = x.clone()
        if self.l_connection and recurrent_input is not None:
            lateral = self.lateral(recurrent_input)
            pre = pre + lateral
        if top_down_input is not None:
            if self.identity_b:
                size_difference = self.output_size - top_down_input.shape[1]
                if size_difference != 0:
                    bottom = x[:, :-size_difference]
                    pre[:, :-size_difference] = bottom + top_down_input
                else:
                    bottom = x
                    pre = bottom + top_down_input          
            else:
                pre = pre + top_down_input
        elif self.identity_b:
            bottom = x
        if self.dropout != 0:
            pre = self.dropout_layer(pre)
        if not self.identity_b and not self.disregard_xy_ReLU:
            y = self.activation_func(pre)
        else:
            y = pre.clone()
            y[:, :-2] = self.activation_func(y[:, :-2])
        if self.t_connection:
            top_down = self.top_down(y)
        else:
            top_down = None
        if not self.disentangled_loss:
            # return activation, top-down and preactivation
            return y, top_down, pre
        else:
            if not self.identity_b:
                if recurrent_input is not None:
                    if top_down_input is not None:
                        return y, pre, torch.cat((self.activation_func(pre), bottom, lateral, top_down_input), dim=1)
                    else:
                        return y, pre, torch.cat((self.activation_func(pre), bottom, lateral), dim=1)
                else:
                    if top_down_input is not None:
                        return y, pre, torch.cat((self.activation_func(pre), bottom, top_down_input), dim=1)
                    else:
                        return y, pre, torch.cat((self.activation_func(pre), bottom), dim=1)
            else:
                if top_down_input is None:
                    return y, pre, torch.zeros_like(self.activation_func(pre))
                else:
                    return y, pre, torch.cat((self.activation_func(pre), top_down_input), dim=1)
    

class LSTM_block(torch.nn.Module):
    """
    A RNN block that gets bottom up, has a lateral connection and outputs a top-down connection.
    Blocks can be stacked on top of each other (allows for top-down input as well)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=None,
                 b_connection=True,
                 l_connection=False,
                 t_connection=True,
                 excitatory_b=False,
                 identity_b = False,
                 activation_func=None):
        super(LSTM_block, self).__init__()
        self.activation_func = activation_func
        self.b_connection = b_connection
        self.l_connection = l_connection
        self.t_connection = t_connection
        if output_size is None:
            self.output_size = input_size
        else:
            self.output_size = output_size
        self.input_size = input_size
        if b_connection and not identity_b:
            self.bottom_up = torch.nn.Linear(input_size, hidden_size)
        if l_connection:
            self.lateral = torch.nn.Linear(hidden_size, hidden_size)
        if t_connection:
            self.top_down = torch.nn.Linear(hidden_size, self.output_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size)
        self.identity_b = identity_b
        self.excitatory_b = excitatory_b
        if excitatory_b:
            class WeightClipper(object):
                def __init__(self, frequency=5):
                    self.frequency = frequency
                def __call__(self, module):
                    if hasattr(module, 'weight'):
                        w = module.weight.data
                        w = w.clamp(0,1)
            clipper = WeightClipper()
            self.bottom_up.apply(clipper)
    
    def forward(self, x, recurrent_input=None, top_down_input=None):
        if recurrent_input is not None:
            lateral_input, lstm_input = recurrent_input
        else:
            lateral_input = None
            lstm_input = None
        if self.b_connection and not self.identity_b:
            x = self.bottom_up(x)
        if self.l_connection and lateral_input is not None:
            x = x + self.lateral(lateral_input)
        if top_down_input is not None:
            if self.identity_b:
                size_difference = self.output_size - top_down_input.shape[1]
                if size_difference != 0:
                    x[:, :-size_difference] = x[:, :-size_difference] + top_down_input
                else:
                    x = x + top_down_input
            else:
                x = x + top_down_input
        if lstm_input is None:
            x, lstm_input = self.lstm(x)
        else:
            x, lstm_input = self.lstm(x, lstm_input)
        if self.t_connection:
            y = self.top_down(x)
        else:
            y = self.activation_func(x)
        return y, x, lstm_input



class RNN_block_conv(torch.nn.Module):
    """
    A convolutional RNN block that gets bottom up, has a lateral connection and outputs a top-down connection.
    Blocks can be stacked on top of each other (allows for top-down input as well)
    """

    def __init__(self,
                 input_channel,
                 hidden_channel,
                 kernel_size,
                 output_channel=None,
                 b_connection=True,
                 l_connection=True,
                 t_connection=True,
                 excitatory_b=False,
                 identity_b = False,
                 use_activation_func=True,
                 activation_func=None,
                 pool_input=False):
        super(RNN_block_conv, self).__init__()
        self.b_connection = b_connection
        self.l_connection = l_connection
        self.t_connection = t_connection
        self.use_activation_func = use_activation_func
        self.activation_func = activation_func
        self.pool_input = pool_input
        if output_channel is None:
            self.output_channel = input_channel
        else:
            self.output_channel = output_channel
        self.input_channel = input_channel
        if b_connection and not identity_b:
            self.bottom_up = torch.nn.Conv2d(input_channel, hidden_channel, kernel_size, padding='same')
        if l_connection:
            self.lateral = torch.nn.Conv2d(hidden_channel, hidden_channel, kernel_size, padding='same')
        if t_connection:
            padding = int((kernel_size-1)/2)
            self.top_down = torch.nn.ConvTranspose2d(hidden_channel, self.output_channel, kernel_size, stride=2, padding=padding, output_padding=1)
        if pool_input:
            self.pool = torch.nn.MaxPool2d(2, 2)
        self.identity_b = identity_b
        self.excitatory_b = excitatory_b
        if excitatory_b:
            class WeightClipper(object):
                def __init__(self, frequency=5):
                    self.frequency = frequency
                def __call__(self, module):
                    if hasattr(module, 'weight'):
                        w = module.weight.data
                        w = w.clamp(0,1)
            clipper = WeightClipper()
            self.bottom_up.apply(clipper)
    
    def forward(self, x, recurrent_input=None, top_down_input=None):
        if self.pool_input:
            x = self.pool(x)
        if self.b_connection and not self.identity_b:
            x = self.bottom_up(x)
            if self.use_activation_func:
                x = self.activation_func(x)
        if self.l_connection and recurrent_input is not None:
            x = x + self.lateral(recurrent_input)
        if top_down_input is not None:
            if self.identity_b:
                channel_difference = self.output_channel - top_down_input.shape[1]
                if channel_difference != 0:
                    x[:, :-channel_difference] = x[:, :-channel_difference] + top_down_input
                else:
                    x = x + top_down_input
            else:
                x = x + top_down_input
        if self.t_connection:
            top_down = self.top_down(x)
        else:
            top_down = None
        if not self.identity_b:
            y = self.activation_func(x)
        else:
            y = x.clone()
            y[:, :-2] = self.activation_func(y[:, :-2])
        # return activation, top-down and preactivation
        return y, top_down, x


class State(ModelState):
    """
    A Wrapper for the RNN, handling the splitting of input sequences into time-steps as well as conveying of hidden states between time-steps, loss calculation
    and extraction of activations and predictions of the model.
    """

    def __init__(self,
                 activation_func,
                 optimizer,
                 lr: float,
                 title: str,
                 input_size: int,
                 hidden_size: int,
                 device: str,
                 deterministic=True,
                 weights_init=functions.init_params,
                 prevbatch=False,
                 conv=False,
                 use_fixation=True,
                 seed=None,
                 use_conv=False,
                 use_lstm=False,
                 warp_imgs=True, 
                 use_resNet=False,
                 time_steps_img=1,
                 time_steps_cords=1,
                 mnist=False,
                 twolayer=True,
                 dropout=0,
                 disentangled_loss=False,
                 useReservoir=False):
        self.mnist = mnist
        if seed != None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.seed = seed
        if self.mnist:
            self.foveal_transform = FovealTransform(fovea_size=0.1, img_target_size=54, img_size=(140,56), jitter_type=None, jitter_amount=0, device=device, warp_imgs=warp_imgs)
        else:
            if input_size == 85*85:
                self.foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=85, img_size=(256,256), jitter_type=None, jitter_amount=0, device=device, warp_imgs=warp_imgs)
            else:
                self.foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=128, img_size=(256,256), jitter_type=None, jitter_amount=0, device=device, warp_imgs=warp_imgs)
            # self.foveal_transform = FovealTransform(fovea_size=0.2, img_target_size=85, img_size=(256,256), jitter_type=None, jitter_amount=0, device=device, warp_imgs=warp_imgs)

        ModelState.__init__(self,

                            ResNetRNN(input_size, hidden_size, activation_func, device=device, use_fixation=use_fixation, use_conv=use_conv, use_lstm=use_lstm, use_resNet=use_resNet, time_steps_img=time_steps_img, time_steps_cords=time_steps_cords, twolayer=twolayer, dropout=dropout, disentangled_loss=disentangled_loss, useReservoir=useReservoir).to(device),
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
                            device)

    def run(self, batch, fixations, loss_fn, state=None):
        """
        Runs a batch of sequences through the model

        Returns:
            loss,
            training metadata
        """
        batch = batch.to(self.device)
        fixations = fixations.to(self.device)
        if len(batch.shape) == 2:
            batch_size = batch.shape[1]
            batch = batch.permute(1,0).reshape(batch_size, 1, 140, 56)
        else:
            batch_size = batch.shape[0]
            if len(batch.shape) > 3:
                batch = batch.reshape(batch_size, 3, 256, 256)
            else:
                batch = batch.reshape(batch_size, 1, 256, 256)
        h = self.model.init_state(batch_size)
        loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
        loss = loss.to(self.device)

        # fixations are (batch_size, seq_len, 2)
        if len(fixations.shape) == 2:
            fixations = fixations.permute(1,0).reshape(batch_size, 10, 2)
        images = self.foveal_transform(batch, fixations)
        if len(images.shape) == 3:
            images = images.permute(1, 0, 2)
        else:
            images = images.permute(1, 0, 2, 3, 4)

        if not self.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
            if self.mnist:
                fixations[:, :, 1] = fixations[:, :, 1] / 0.4
        recurrent_state = None
        for i, image in enumerate(images):
            if self.model.use_conv:
                image = image.reshape(image.shape[0], 128, 128)
            for t in range(self.model.time_steps_img):
                if t >= self.model.time_steps_img - self.model.time_steps_cords:
                    h, l_a, recurrent_state = self.model(image, fixation=fixations[:, i], state=h, recurrent_state=recurrent_state)  # l_a is now a list of potential loss terms
                    loss = loss + self.loss(l_a, loss_fn)
                    # if i == 6 and t == 5:
                    #     print(f"{t}, {self.loss(l_a, loss_fn)}")
                else:
                    h, l_a, recurrent_state = self.model(image, fixation=torch.zeros_like(fixations[:, i]), state=h, recurrent_state=recurrent_state)  # l_a is now a list of potential loss terms
                    # if t==0:
                    loss = loss + self.loss(l_a, loss_fn)
                    # if i == 6 and t == 0:
                    #     print(f"{t}, {self.loss(l_a, loss_fn)}")
        return loss, loss.detach(), None

    def loss(self, loss_terms, loss):
        loss_t1, loss_t2, beta = loss, None, 1
        # split for weighting
        if 'beta' in loss:
            beta, loss = loss.split('beta')
            beta = float(beta)
        if 'and' in loss:
            loss_t1, loss_t2 = loss.split('and')

        # parse loss terms
        loss_fn_t1, loss_arg_t1 = functions.parse_loss(loss_t1, loss_terms)
        loss_fn_t2, loss_arg_t2 = functions.parse_loss(loss_t2, loss_terms)

        # return loss_fn_t1(loss_arg_t1) + beta * loss_fn_t2(loss_arg_t2)
        return loss_fn_t1(loss_arg_t1)

    def predict(self, x, fixation, recurrent_state=None):
        """
        Returns the networks 'prediction' for the input.
        """
        if self.model.use_resNet:
            x = self.model.flatten(self.model.resNet(x))
        h = self.model.init_state(x.shape[0])
        if self.model.use_conv:
            h = h.reshape(h.shape[0], 128, 128)
        h = h.to(self.device)
        hidden_states = []
        next_top_down_states = []
        if self.model.use_fixation:
            if not self.model.use_conv:
                if self.model.use_grid_coding:
                    grid_code = self.model.grid_cell_layer(fixation)
                    x = torch.cat((x, grid_code), 1)
                else:
                    x = torch.cat((x, fixation), 1)
            elif not self.model.hybrid:
                fixation_x = fixation[:, 0].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                fixation_y = fixation[:, 1].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                x = torch.stack([x, fixation_x, fixation_y], 1)
            else:
                x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        if recurrent_state is None:
            recurrent_states = [None] * len(self.model.layers)
            top_down_states = [None] * len(self.model.layers)
            for i, layer in enumerate(self.model.layers):
                if i + 1 < len(self.model.layers):
                    t_state = top_down_states[i+1]
                else:
                    t_state = None
                if i == 0:
                    if self.model.use_fixation:
                        if not self.model.use_conv:
                            if self.model.use_grid_coding:
                                grid_code = self.model.grid_cell_layer(fixation)
                                h = torch.cat((h, grid_code), 1)
                            else:
                                h = torch.cat((h, fixation), 1)
                            y, top_down, _ = layer(h, recurrent_states[i], t_state)
                            if self.model.relay_image:
                                y = h
                        elif not self.model.hybrid:
                            fixation_x = fixation[:, 0].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                            fixation_y = fixation[:, 1].repeat(128 * 128).reshape(128, 128, fixation.shape[0]).permute(2, 0, 1)
                            h = torch.stack([h, fixation_x, fixation_y], 1)
                            y, top_down, _ = layer(h, recurrent_states[i], t_state)
                        else:
                            h = h.reshape(h.shape[0], 1, h.shape[1], h.shape[2])
                            y, top_down, _ = layer(h, recurrent_states[i], t_state)
                    else:
                        y, top_down, _ = layer(h, recurrent_states[i], t_state)
                elif self.model.use_conv and self.model.hybrid and i == self.model.hybrid_point:
                    shape = y.shape
                    y = y.reshape(y.shape[0], y.shape[1] * y.shape[2] * y.shape[3])
                    y, top_down, _ = layer(torch.cat((y, fixation), 1), recurrent_states[i], t_state)
                    top_down = top_down.reshape(shape)
                elif self.model.use_lstm:
                    top_down, y, rec_state = layer(y, recurrent_states[i], t_state)
                else:
                    y, top_down, _ = layer(y, recurrent_states[i], t_state)
                if self.model.lesion and i != 0:
                # print(x[:, self.lesion_map[(i-1)*2]].shape)
                    y[:, self.model.lesion_map[(i-1)*2]] = 0
                    y[:, self.model.lesion_map[(i-1)*2+1]] = 0
                if not self.model.use_lstm or i==0:
                    hidden_states.append(y)
                else:
                    hidden_states.append((y, rec_state))
                next_top_down_states.append(top_down)
            recurrent_state = [hidden_states, next_top_down_states]
        hidden_states = []
        next_top_down_states = []
        recurrent_states = recurrent_state[0]
        top_down_states = recurrent_state[1]
        for i, layer in enumerate(self.model.layers):
            if i + 1 < len(self.model.layers):
                t_state = top_down_states[i+1]
            else:
                t_state = None
            if self.model.use_conv and self.model.hybrid and i == self.model.hybrid_point:
                shape = x.shape
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
                x ,top_down, _ = layer(torch.cat((x, fixation), 1), recurrent_states[i], t_state)
                top_down = top_down.reshape(shape)
            else:
                if not self.model.use_lstm or i==0:
                    z, top_down, _ = layer(x, recurrent_states[i], t_state)
                else:
                    top_down, z, rec_state = layer(x, recurrent_states[i], t_state)
                if not self.model.relay_image or not i == 0:
                    x = z
            if self.model.lesion and i != 0:
                # print(x[:, self.lesion_map[(i-1)*2]].shape)
                x[:, self.model.lesion_map[(i-1)*2]] = 0
                x[:, self.model.lesion_map[(i-1)*2+1]] = 0
            if self.model.relay_image:
                hidden_states.append(z)
            elif self.model.use_lstm and not i==0:
                hidden_states.append((x, rec_state))
            else:
                hidden_states.append(x)
            next_top_down_states.append(top_down)
        pred = None
        if top_down_states is not None and len(top_down_states) > 1:
            pred = top_down_states[1].squeeze()
        if self.model.layer1.l_connection:
            if recurrent_states[0] is not None:
                if pred is not None:
                    if len(pred.shape) == 1:
                        pred += self.model.layer1.lateral(recurrent_states[0])[0, :pred.shape[0]]
                    else:
                        pred += self.model.layer1.lateral(recurrent_states[0])[:, :pred.shape[1]]
                else:
                    pred = self.model.layer1.lateral(recurrent_states[0]).squeeze()
        return pred, [hidden_states, next_top_down_states]
        

    def get_preactivations(self, batch, fixations, layer=0, timestep=None):
        """
        Runs a batch of sequences through the model

        Returns:
            list of preactivations of the layers
        """
        activations = None
        batch = batch.to(self.device)
        fixations = fixations.to(self.device)
        if len(batch.shape) == 2:
            batch_size = batch.shape[1]
            batch = batch.permute(1,0).reshape(batch_size, 1, 140, 56)
        else:
            batch_size = batch.shape[0]
            if len(batch.shape) > 3:
                batch = batch.reshape(batch_size, 3, 256, 256)
            else:
                batch = batch.reshape(batch_size, 1, 256, 256)
        h = self.model.init_state(batch_size)
        loss = torch.zeros(1, dtype=torch.float, requires_grad=True)
        loss = loss.to(self.device)

        # fixations are (batch_size, seq_len, 2)
        if len(fixations.shape) == 2:
            fixations = fixations.permute(1,0).reshape(batch_size, 10, 2)
        images = self.foveal_transform(batch, fixations)
        if len(images.shape) == 3:
            images = images.permute(1, 0, 2)
        else:
            images = images.permute(1, 0, 2, 3, 4)
        if not self.model.use_grid_coding:
            for i in range(fixations.shape[1]):
                if i == fixations.shape[1]-1:
                    fixations[:, i] = fixations[:, i] - fixations[:, i]
                else:
                    fixations[:, i] = fixations[:, i+1] - fixations[:, i]
            if self.mnist:
                fixations[:, :, 1] = fixations[:, :, 1] / 0.4
        recurrent_state = None
        for i, image in enumerate(images):
            if self.model.use_conv:
                image = image.reshape(image.shape[0], 128, 128)
            for t in range(self.model.time_steps_img):
                if t >= self.model.time_steps_img - self.model.time_steps_cords:
                    h, l_a, recurrent_state = self.model(image, fixation=fixations[:, i], state=h, recurrent_state=recurrent_state)  # l_a is now a list of potential loss terms
                    if self.model.use_lstm:
                        activations = torch.cat((activations, l_a[1][0][layer].unsqueeze(1)), dim=1)
                    else:
                        if timestep is None or timestep == t:
                            if activations is None:
                                activations = l_a[1][layer].unsqueeze(1)
                            else:
                                activations = torch.cat((activations, l_a[1][layer].unsqueeze(1)), dim=1)
                else:
                    h, l_a, recurrent_state = self.model(image, fixation=torch.zeros_like(fixations[:, i]), state=h, recurrent_state=recurrent_state)  # l_a is now a list of potential loss terms
                    if self.model.use_lstm:
                        if i==0 and t==0:
                            activations = l_a[1][0][layer].unsqueeze(1)
                        else:
                            activations = torch.cat((activations, l_a[1][0][layer].unsqueeze(1)), dim=1)
                    else:
                        if timestep is None or timestep == t:
                            if activations is None:
                                activations = l_a[1][layer].unsqueeze(1)
                            else:
                                activations = torch.cat((activations, l_a[1][layer].unsqueeze(1)), dim=1)
        return activations

    def get_activations(self, batch, fixations, layer=0, timestep=None):
        """
        Runs a batch of sequences through the model

        Returns:
            List of activations of the layers
        """
        preactivations = self.get_preactivations(batch, fixations, layer, timestep)
        return self.model.activation_func(preactivations)



    def step(self, loss, scaler=None):
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            loss.backward()
            # print(torch.norm(torch.cat([p.grad.detach().flatten() for p in self.model.parameters()]), 2.0).item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            # print(torch.norm(torch.cat([p.grad.detach().flatten() for p in self.model.parameters()]), 2.0).item())

            self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
