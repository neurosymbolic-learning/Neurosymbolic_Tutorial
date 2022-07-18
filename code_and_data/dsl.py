#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Neural Functions

# In[ ]:


def init_neural_function(input_type, output_type, input_size, output_size, num_units, device="cpu"):
    if (input_type, output_type) == ("list", "list"):
        return ListToListModule(input_size, output_size, num_units, device=device)
    elif (input_type, output_type) == ("list", "atom"):
        return ListToAtomModule(input_size, output_size, num_units, device=device)
    elif (input_type, output_type) == ("atom", "atom"):
        return AtomToAtomModule(input_size, output_size, num_units, device=device)
    else:
        raise NotImplementedError


class HeuristicNeuralFunction:

    def __init__(self, input_type, output_type, input_size, output_size, num_units, name, device):
        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_units = num_units
        self.name = name
        self.device = device
        
        self.init_model()

    def init_model(self):
        raise NotImplementedError

class ListToListModule(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        super().__init__("list", "list", input_size, output_size, num_units, "ListToListModule", device=device)

    def init_model(self):
        self.model = RNNModule(self.input_size, self.output_size, self.num_units).to(self.device)

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        model_out = self.model(batch, batch_lens)
        assert len(model_out.size()) == 3
        return model_out

class ListToAtomModule(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        super().__init__("list", "atom", input_size, output_size, num_units, "ListToAtomModule", device=device)

    def init_model(self):
        self.model = RNNModule(self.input_size, self.output_size, self.num_units, device=self.device).to(device=self.device)

    def execute_on_batch(self, batch, batch_lens, is_sequential=False):
        # print(batch.size())
        assert len(batch.size()) == 3
        model_out = self.model(batch, batch_lens)
        assert len(model_out.size()) == 3

        if not is_sequential:
            idx = torch.tensor(batch_lens).to(self.device) - 1
            idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, model_out.size(-1))
            model_out = model_out.gather(1, idx).squeeze(1)

        return model_out

class ListToAtomModuleConv1D(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units, seq_len = 13, device="cpu"):
        self.seq_len = seq_len
        super().__init__("list", "atom", input_size, output_size, num_units, "ListToAtomModuleConv1D", device=device)

    def __call__(self, batch):
        return self.execute_on_batch(batch, None)

    def init_model(self):
        # in_channels and out_channels are the same size. This ensures that
        # a separate set of weights is learned per feature
        self.model = nn.Conv1d(self.input_size, self.input_size * self.output_size, self.seq_len,
                               groups=self.input_size).to(self.device)

    def execute_on_batch(self, batch, batch_lens, is_sequential=False):
        assert len(batch.size()) == 3
        # batch is batch_size, seq_len, input_size, but need to pass in
        # batch_size, input_size, seq_len
        # print(batch.size())
        batch = batch.transpose(1, 2)
        # output is batch_size, input_size * output_size, 1
        model_out = self.model(batch).squeeze().view(-1, self.input_size, self.output_size)
        model_out = torch.sum(model_out, dim=1)
        return model_out.view(-1, self.output_size)

class AtomToAtomModule(HeuristicNeuralFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        super().__init__("atom", "atom", input_size, output_size, num_units, "AtomToAtomModule", device=device)

    def init_model(self):
        self.model = FeedForwardModule(self.input_size, self.output_size, self.num_units, device=self.device)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        model_out = self.model(batch)
        assert len(model_out.size()) == 2
        return model_out


##############################
####### NEURAL MODULES #######
##############################


class RNNModule(nn.Module):

    def __init__(self, input_size, output_size, num_units, num_layers=1, device="cpu"):
        super(RNNModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn_size = num_units
        self.num_layers = num_layers
        self.rnn = nn.LSTM(self.input_size, self.rnn_size, num_layers=self.num_layers).to(device)
        self.out_layer = nn.Linear(self.rnn_size, self.output_size)
        self.device = device

    def init_hidden(self, batch_size):
        ahid = torch.zeros(self.num_layers, batch_size, self.rnn_size)
        bhid = torch.zeros(self.num_layers, batch_size, self.rnn_size)
        ahid = ahid.requires_grad_(True)
        bhid = bhid.requires_grad_(True)
        hid = (ahid.to(self.device), bhid.to(self.device))
        return hid

    def forward(self, batch, batch_lens):
        assert isinstance(batch, torch.Tensor)
        batch_size, seq_len, feature_dim = batch.size()

        # pass through rnn
        hidden = self.init_hidden(batch_size)
        batch_packed = torch.nn.utils.rnn.pack_padded_sequence(batch, batch_lens, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(batch_packed, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # pass through linear layer
        out = out.contiguous()
        out = out.view(-1, out.shape[2])
        out = self.out_layer(out)
        out = out.view(batch_size, seq_len, -1)

        return out

class FeedForwardModule(nn.Module):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        super(FeedForwardModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = num_units
        self.first_layer = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.out_layer = nn.Linear(self.hidden_size, self.output_size).to(device)
        self.device = device

    def forward(self, current_input):
        assert isinstance(current_input, torch.Tensor)
        current_input = current_input.to(self.device)
        current = F.relu(self.first_layer(current_input))
        current = self.out_layer(current)
        return current


# ## Generic Library Functions

# In[ ]:


class LibraryFunction:

    def __init__(self, submodules, input_type, output_type, input_size, 
                 output_size, num_units, name="", has_params=False, device="cpu"):
        self.submodules = submodules
        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.num_units = num_units
        self.name = name
        self.has_params = has_params
        self.device = device

        if self.has_params:
            assert "init_params" in dir(self)
            self.init_params()

    def get_submodules(self):
        return self.submodules

    def set_submodules(self, new_submodules):
        self.submodules = new_submodules

    def get_typesignature(self):
        return self.input_type, self.output_type

class StartFunction(LibraryFunction):

    def __init__(self, input_type, output_type, input_size, output_size, num_units, device="cpu"):
        self.program = init_neural_function(input_type, output_type, input_size, output_size, num_units, device=device)
        submodules = { 'program' : self.program }
        super().__init__(submodules, input_type, output_type, input_size, 
                         output_size, num_units, name="Start", device=device)

    def execute_on_batch(self, batch, batch_lens=None, batch_output=None, is_sequential=False):
        return self.submodules["program"].execute_on_batch(batch, batch_lens)
            
class FoldFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, fold_function=None, device="cpu"):
        #will this accumulator require a grad?
        self.accumulator = torch.zeros(output_size)
        if fold_function is None:
            fold_function = init_neural_function("atom", "atom", input_size+output_size, output_size, num_units, device=device)
        submodules = { "foldfunction" : fold_function }
        super().__init__(submodules, "list", "atom", input_size, output_size, num_units, name="Fold", device=device)

    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        batch = batch.transpose(0,1) # (seq_len, batch_size, feature_dim)

        fold_out = []
        folded_val = self.accumulator.clone().detach().requires_grad_(True)
        folded_val = folded_val.unsqueeze(0).repeat(batch_size,1).to(self.device)
        for t in range(seq_len):
            features = batch[t]
            out_val = self.submodules["foldfunction"].execute_on_batch(torch.cat([features, folded_val], dim=1))
            fold_out.append(out_val.unsqueeze(1))
            folded_val = out_val
        fold_out = torch.cat(fold_out, dim=1)
        
        if not is_sequential:
            idx = torch.tensor(batch_lens).to(self.device) - 1
            idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, fold_out.size(-1))
            fold_out = fold_out.gather(1, idx).squeeze(1)

        return fold_out

class MapFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, map_function=None, device="cpu"):
        if map_function is None:
            map_function = init_neural_function("atom", "atom", input_size, output_size, num_units, device=device)
        submodules = { "mapfunction" : map_function }
        super().__init__(submodules, "list", "list", input_size, output_size, num_units, name="Map", device=device)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        map_input = batch.view(-1, feature_dim)
        map_output = self.submodules["mapfunction"].execute_on_batch(map_input)
        return map_output.view(batch_size, seq_len, -1)

class MapPrefixesFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, map_function=None, device="cpu"):
        if map_function is None:
            map_function = init_neural_function("list", "atom", input_size, output_size, num_units, device=device)
        submodules = { "mapfunction" : map_function }
        super().__init__(submodules, "list", "list", input_size, output_size, num_units, name="MapPrefixes", device=device)

    def execute_on_batch(self, batch, batch_lens):
        assert len(batch.size()) == 3
        map_output = self.submodules["mapfunction"].execute_on_batch(batch, batch_lens, is_sequential=True)
        assert len(map_output.size()) == 3
        return map_output

class ITE(LibraryFunction):
    """(Smoothed) If-The-Else."""

    def __init__(self, input_type, output_type, input_size, output_size, num_units, 
                 eval_function=None, function1=None, function2=None, beta=1.0, name="ITE", simple=False, device="cpu"):
        if eval_function is None:
            if simple:
                eval_function = init_neural_function(input_type, "atom", input_size, 1, num_units, device=device)
            else:
                eval_function = init_neural_function(input_type, "atom", input_size, output_size, num_units, device=device)
        if function1 is None:
            function1 = init_neural_function(input_type, output_type, input_size, output_size, num_units, device=device)
        if function2 is None:
            function2 = init_neural_function(input_type, output_type, input_size, output_size, num_units, device=device)
        submodules = { "evalfunction" : eval_function, "function1" : function1, "function2" : function2 }
        self.bsmooth = nn.Sigmoid()
        self.beta = beta
        self.simple = simple # the simple version of ITE evaluates the same function for all dimensions of the output
        super().__init__(submodules, input_type, output_type, input_size, output_size, num_units, name=name, device=device)

    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        if self.input_type == 'list':
            assert len(batch.size()) == 3
            assert batch_lens is not None
        else:
            assert len(batch.size()) == 2
        if is_sequential:
            predicted_eval = self.submodules["evalfunction"].execute_on_batch(batch, batch_lens, is_sequential=False)
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens, is_sequential=is_sequential)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens, is_sequential=is_sequential)
        else:
            predicted_eval = self.submodules["evalfunction"].execute_on_batch(batch, batch_lens)
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens)

        gate = self.bsmooth(predicted_eval*self.beta)
        if self.simple:
            gate = gate.repeat(1, self.output_size)
        
        if self.get_typesignature() == ('list', 'list'):
            gate = gate.unsqueeze(1).repeat(1, batch.size(1), 1)
        elif self.get_typesignature() == ('list', 'atom') and is_sequential:
            gate = gate.unsqueeze(1).repeat(1, batch.size(1), 1)

        assert gate.size() == predicted_function2.size() == predicted_function1.size()
        ite_result = gate*predicted_function1 + (1.0 - gate)*predicted_function2

        return ite_result

class SimpleITE(ITE):
    """The simple version of ITE evaluates one function for all dimensions of the output."""

    def __init__(self, input_type, output_type, input_size, output_size, num_units, eval_function=None, function1=None, function2=None, beta=1.0, device="cpu"):
        super().__init__(input_type, output_type, input_size, output_size, num_units, 
            eval_function=eval_function, function1=function1, function2=function2, beta=beta, name="SimpleITE", simple=True, device=device)
        
class AndFunction(LibraryFunction):

    def __init__(self, input_type, output_type, input_size, output_size, num_units, function1=None, function2=None, device="cpu"):
        if function1 is None:
            function1 = init_neural_function(input_type, output_type, input_size, output_size, num_units, device=device)
        if function2 is None:
            function2 = init_neural_function(input_type, output_type, input_size, output_size, num_units, device=device)
        submodules = { "function1" : function1, "function2" : function2 }
        super().__init__(submodules, input_type, output_type, input_size, output_size, num_units, name="And", device=device)

    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        if self.input_type == 'list':
            assert len(batch.size()) == 3
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens=batch_lens,
                                                                                is_sequential=is_sequential)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens=batch_lens,
                                                                                is_sequential=is_sequential)
        else:
            assert len(batch.size()) == 2
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens=batch_lens)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens=batch_lens)
        return predicted_function1 * predicted_function2

class OrFunction(LibraryFunction):

    def __init__(self, input_type, output_type, input_size, output_size, num_units, function1=None, function2=None, device="cpu"):
        if function1 is None:
            function1 = init_neural_function(input_type, output_type, input_size, output_size, num_units, device=device)
        if function2 is None:
            function2 = init_neural_function(input_type, output_type, input_size, output_size, num_units, device=device)
        submodules = { "function1": function1, "function2": function2 }
        super().__init__(submodules, input_type, output_type, input_size, output_size, num_units, name="Or", device=device)

    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        if self.input_type == 'list':
            assert len(batch.size()) == 3
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens=batch_lens,
                                                                                is_sequential=is_sequential)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens=batch_lens,
                                                                                is_sequential=is_sequential)
        else:
            assert len(batch.size()) == 2
            predicted_function1 = self.submodules["function1"].execute_on_batch(batch, batch_lens=batch_lens)
            predicted_function2 = self.submodules["function2"].execute_on_batch(batch, batch_lens=batch_lens)
        return predicted_function1 + predicted_function2

class ContinueFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, fxn=None, device="cpu"):
        if fxn is None:
            fxn = init_neural_function("atom", "atom", input_size, output_size, num_units, device=device)
        submodules = { "fxn" : fxn }
        super().__init__(submodules, "atom", "atom", input_size, output_size, num_units, name="", device=device)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        fxn_out = self.submodules["fxn"].execute_on_batch(batch)
        return fxn_out

class LearnedConstantFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        super().__init__({}, "atom", "atom", input_size, output_size, num_units, name="LearnedConstant", has_params=True, device=device)

    def init_params(self):
        self.parameters = { "constant" : torch.rand(self.output_size, requires_grad=True, device=self.device) }

    def execute_on_batch(self, batch, batch_lens=None):
        return self.parameters["constant"].unsqueeze(0).repeat(batch.size(0), 1)
        
class AffineFunction(LibraryFunction):

    def __init__(self, raw_input_size, selected_input_size, output_size, num_units, name="Affine", device="cpu"):
        self.selected_input_size = selected_input_size
        super().__init__({}, "atom", "atom", raw_input_size, output_size, 
                         num_units, name=name, has_params=True, device=device)

    def init_params(self):
        self.linear_layer = nn.Linear(self.selected_input_size, self.output_size, bias=True).to(self.device)
        self.parameters = {
            "weights" : self.linear_layer.weight,
            "bias" : self.linear_layer.bias
        }

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        return self.linear_layer(batch)

class AffineFeatureSelectionFunction(AffineFunction):

    def __init__(self, input_size, output_size, num_units, name="AffineFeatureSelection", device="cpu"):
        assert hasattr(self, "full_feature_dim")
        assert input_size >= self.full_feature_dim
        if self.full_feature_dim == 0:
            self.is_full = True
            self.full_feature_dim = input_size
        else:
            self.is_full = False
        additional_inputs = input_size - self.full_feature_dim

        assert hasattr(self, "feature_tensor")
        assert len(self.feature_tensor) <= input_size
        self.feature_tensor = self.feature_tensor.to(device)
        super().__init__(raw_input_size=input_size, selected_input_size=self.feature_tensor.size()[-1]+additional_inputs, 
            output_size=output_size, num_units=num_units, name=name, device=device)

    def init_params(self):
        self.raw_input_size = self.input_size
        if self.is_full:
            self.full_feature_dim = self.input_size
            self.feature_tensor = torch.arange(self.input_size).to(self.device)

        additional_inputs = self.raw_input_size - self.full_feature_dim
        self.selected_input_size = self.feature_tensor.size()[-1] + additional_inputs
        self.linear_layer = nn.Linear(self.selected_input_size, self.output_size, bias=True).to(self.device)
        self.parameters = {
            "weights" : self.linear_layer.weight,
            "bias" : self.linear_layer.bias
        }

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        features = torch.index_select(batch, 1, self.feature_tensor)
        remaining_features = batch[:,self.full_feature_dim:]
        return self.linear_layer(torch.cat([features, remaining_features], dim=-1))

class NeuralFeatureSelectionFunction(LibraryFunction):

    def __init__(self, input_size, output_size, num_units, name="NeuralFeatureSelection", device="cpu"):

        self.feature_tensor = torch.arange(input_size)
        self.feature_tensor = self.feature_tensor.to(device)
        self.custom_nn = True
        hidden_dim = 32

        self.model = nn.Sequential(
               nn.Linear(input_size,hidden_dim),
               nn.ReLU(),
               nn.Linear(hidden_dim,output_size)).to(device)

        # def __init__(self, submodules, input_type, output_type, input_size, 
        #              output_size, num_units, name="", has_params=False, device="cpu"):

        super().__init__({}, 'atom', 'atom', input_size, output_size, 
                         num_units=num_units, name=name, device=device)

    def execute_on_batch(self, batch, batch_lens=None):
        assert len(batch.size()) == 2
        features = torch.index_select(batch, 1, self.feature_tensor)

        return self.model(features)

class FullInputAffineFunction(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = 0 # this will indicate additional_inputs = 0 in FeatureSelectionFunction
        self.feature_tensor = torch.arange(input_size) # selects all features by default
        super().__init__(input_size, output_size, num_units, name="FullFeatureSelect", device=device)


class RunningAverageFunction(LibraryFunction):
    """Computes running average over a window, then applies an Atom2AtomModule on the average."""

    def __init__(self, input_size, output_size, num_units, a2a_function=None, name="RunningAvg", device="cpu"):
        self.device = device

        if a2a_function is None:
            a2a_function = init_neural_function("atom", "atom", input_size, output_size, num_units, device = device)
        submodules = { "subfunction" : a2a_function }
        super().__init__(submodules, "list", "atom", input_size, output_size, num_units, name = name, device = device)

    def window_start(self, t):
        return 0

    def window_end(self, t):
        return t

    def execute_on_batch(self, batch, batch_lens, is_sequential=False):
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        batch = batch.transpose(0,1) # (seq_len, batch_size, feature_dim)

        out = []
        for t in range(seq_len):
            window_start = max(0, self.window_start(t))
            window_end = min(seq_len, self.window_end(t))
            window = batch[window_start:window_end+1]
            running_average = torch.mean(window, dim=0)
            out_val = self.submodules["subfunction"].execute_on_batch(running_average, 
                                                                      batch_lens = batch_lens)
            out.append(out_val.unsqueeze(1))
        out = torch.cat(out, dim=1)        
        
        if not is_sequential:
            idx = torch.tensor(batch_lens).to(self.device) - 1
            idx = idx.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out.size(-1))
            out = out.gather(1, idx).squeeze(1)

        return out


class RunningAverageWindow5Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None, device="cpu"):
        super().__init__(input_size, output_size, num_units, a2a_function, 
                         name="Window5Avg", device=device)

    def window_start(self, t):
        return t-2

    def window_end(self, t):
        return t+2

class RunningAverageWindow7Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None, device="cpu"):
        super().__init__(input_size, output_size, num_units, a2a_function, 
                         name="Window7Avg", device=device)

    def window_start(self, t):
        return t-3

    def window_end(self, t):
        return t+3


class RunningAverageWindow13Function(RunningAverageFunction):

    def __init__(self, input_size, output_size, num_units, a2a_function=None, device="cpu"):
        super().__init__(input_size, output_size, num_units, a2a_function, 
                         name="Window13Avg", device=device)

    def window_start(self, t):
        return t-6

    def window_end(self, t):
        return t+6


# ## Behavior Library Functions

# ### Mice Feature Subsets

# In[ ]:


CALMS21_FEATURE_SUBSETS = {
    "res_angle_head_body": torch.arange(0, 2, dtype=torch.long),
    "axis_ratio": torch.arange(2, 4, dtype=torch.long),
    "speed": torch.arange(4, 6, dtype=torch.long),
    "acceleration": torch.arange(6, 8, dtype=torch.long),
    "tangential_velocity": torch.arange(8, 10, dtype=torch.long),
    "rel_angle_social": torch.arange(10, 12, dtype=torch.long),
    "angle_between": torch.arange(12, 13, dtype=torch.long),
    "facing_angle": torch.arange(13, 15, dtype=torch.long),
    "overlap_bboxes": torch.arange(15, 16, dtype=torch.long),
    "area_ellipse_ratio": torch.arange(16, 17, dtype=torch.long),
    "min_res_nose_dist": torch.arange(17, 18, dtype=torch.long)
}

# Indices used in a feature subset
CALMS21_FULL_FEATURE_DIM = 18

class CALMS21ResAngleHeadBodySelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["res_angle_head_body"]
        super().__init__(input_size, output_size, num_units, name="ResAngleHeadBodySelect", device=device)


class CALMS21AxisRatioSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["axis_ratio"]
        super().__init__(input_size, output_size, num_units, name="AxisRatioSelect", device=device)


class CALMS21SpeedSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["speed"]
        super().__init__(input_size, output_size, num_units, name="SpeedSelect", device=device)


class CALMS21TangentialVelocitySelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["tangential_velocity"]
        super().__init__(input_size, output_size, num_units, name="TangentialVelocitySelect", device=device)


class CALMS21AccelerationSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["acceleration"]
        super().__init__(input_size, output_size, num_units, name="AccelerationSelect", device=device)


class CALMS21RelAngleSocialSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["rel_angle_social"]
        super().__init__(input_size, output_size, num_units, name="RelativeSocialAngleSelect", device=device)


class CALMS21AngleBetweenSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["angle_between"]
        super().__init__(input_size, output_size, num_units, name="AngleBetweenSelect", device=device)


class CALMS21FacingAngleSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["facing_angle"]
        super().__init__(input_size, output_size, num_units, name="FacingAngleSelect", device=device)


class CALMS21OverlapBboxesSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["overlap_bboxes"]
        super().__init__(input_size, output_size, num_units, name="OverlapBboxesSelect", device=device)


class CALMS21AreaEllipseRatioSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["area_ellipse_ratio"]
        super().__init__(input_size, output_size, num_units, name="AreaEllipseRatioSelect", device=device)


class CALMS21MinResNoseKeypointDistSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units, device="cpu"):
        self.full_feature_dim = CALMS21_FULL_FEATURE_DIM
        self.feature_tensor = CALMS21_FEATURE_SUBSETS["min_res_nose_dist"]
        super().__init__(input_size, output_size, num_units, name="MinResNoseKeypointDistanceSelect", device=device)


# ### Morlet Filter

# In[ ]:


import numpy as np
from abc import abstractmethod

DEFAULT_RANGE = np.pi

class SymmetricFilterOp(LibraryFunction):
    '''
    Template for an operation that first weighs a sequence of inputs
    by a symmetric filter and passes in the weighted average of the inputs
    to a submodule
    '''
    def __init__(self, input_size, output_size, num_units, function=None, name="SymFilterOp", device="cpu"):
        self.input_size = input_size
        # Default values for the parameters
        if function is None:
            function = init_neural_function("atom", "atom", input_size, output_size, num_units, device=device)
        submodules = {"function": function}
        super().__init__(submodules, "list", "atom", input_size, output_size, num_units,
                         name=name, has_params=True, device=device)

    @abstractmethod
    def init_params(self):
        """Initializes filter parameters"""
        pass
    
    @abstractmethod
    def get_filter(self, xvals):
        """Get filter weights on a sequence of input values"""
        pass

    def get_filter_default_xvals(self, seq_len):
        """Takes in seq_len, the number of linearly-spaced inputs. 
        Get filter weights on xvals linearly spaced in [-DEFAULT_RANGE,
        DEFAULT_RANGE] """
        xvals = torch.linspace(-DEFAULT_RANGE, DEFAULT_RANGE, seq_len).to(self.device)
        filter = self.get_filter(xvals)
        return filter

    def execute_on_batch(self, batch, batch_lens=None, is_sequential=False):
        """Apply the temporal filter to a batch of inputs"""
        assert len(batch.size()) == 3
        batch_size, seq_len, feature_dim = batch.size()
        filter = self.get_filter_default_xvals(seq_len)
        # Repeat a (seq_len, 1) vector to get (seq_len, feature_dim)
        filter = filter.repeat(1, feature_dim)
        # Apply the filter, sum to get (batch_size, feature_dim) and execute
        if not is_sequential:
            input = torch.sum(torch.mul(batch, filter), 1)
        else:
            input = torch.sum(torch.mul(batch, filter), 1).repeat(1, seq_len, 1).view(-1, feature_dim)
        output = self.submodules["function"].execute_on_batch(input)
        return output


class MorletFilterOp(SymmetricFilterOp):
    '''
    The symmetric filter operation where the shape of the filter is defined
    by the morlet filter
    '''

    def __init__(self, input_size, output_size, num_units, function=None, name="MorletFilterOp", device="cpu"):
        super().__init__(input_size, output_size, num_units, function=function,
                         name=name, device=device)

    def init_params(self):
        """Initialize Morlet Filter parameters"""
        self.parameters = {
            "s": torch.tensor(0.5, requires_grad=True, device=self.device),
            "w": torch.tensor(0.5, requires_grad=True, device=self.device)
        }

    def get_mor_filter(self, xvals, s, w):
        """Implements the Morlet Filter by applying the wavelet function 
        psi to xvals 
        
        Make sure to return the results with shape [seq_len, 1] instead 
        of [seq_len]. view() or reshape(), as well as math functions in 
        the torch library, e.g. torch.exp and torch.pow, should be helpful. 
        """
        return (torch.exp(-0.5 * torch.pow(w * xvals / s, 2)) *                 torch.cos(w * xvals)).view(-1, 1)

    def get_filter(self, xvals):
        s = self.parameters["s"]
        w = self.parameters["w"]
        return self.get_mor_filter(xvals, s, w)


class AsymmetricFilterOp(SymmetricFilterOp):
    '''
    Template for an operation that first weighs a sequence of inputs
    by a filter that has an asymmetric shape and passes in the weighted 
    average of the inputs to a submodule
    '''

    def __init__(self, input_size, output_size, num_units, function=None, name="AsymFilterOp", device="cpu"):
        super().__init__(input_size, output_size, num_units, function=function,
                        name=name, device=device)

    @abstractmethod
    def init_params(self):
        """Initializes filter parameters"""
        pass
    
    @abstractmethod
    def get_filter(self, xvals, left=True):
        """Get filter weights corresponding to a sequence of inputs, xvals. 
        
        The shape of the filter depends on whether xvals are 
        on the left side of the filter, i.e. left = True, or not"""
        pass

    def get_filter_default_xvals(self, seq_len):
        """
        Takes in seq_len, the total number of linearly-spaced inputs. 

        Get the left filter weights on (seq_len / 2) linearly-spaced xvals 
        in [-pi, 0]. Get the right filter weights on the remaining 
        linearly-spaced xvals in [0, pi].
        
        Return the left results concatenated with the right results.

        """
        # Implement for asymetric Morlet filter (note filter shape)
        # changes from left = True to left = False
        raise NotImplementedError


class AsymMorletFilterOp(AsymmetricFilterOp, MorletFilterOp):
    def __init__(self, input_size, output_size, num_units, function=None, name="AsymMorletFilterOp", device="cpu"):
        super().__init__(input_size, output_size, num_units, function=function, 
                         name=name, device=device)

    def init_params(self):
        """s1 and w1 are parameters for the left side of the filter, s2
        and w2 are for the right side of the filter """
        self.parameters = {
            "s1": torch.tensor(0.5, requires_grad=True, device=self.device),
            "w1": torch.tensor(0.5, requires_grad=True, device=self.device),
            "s2": torch.tensor(0.5, requires_grad=True, device=self.device),
            "w2": torch.tensor(0.5, requires_grad=True, device=self.device)
        }

    def get_filter(self, xvals, left=True):
        s = self.parameters["s1"]
        w = self.parameters["w1"]
        if not left:
            s = self.parameters["s2"]
            w = self.parameters["w2"]
        return self.get_mor_filter(xvals, s, w)


# #### Test Morlet Filter
# 
# To check your implementation in the Morlet Filter section, run  the below tests, which will output 3 plots. Check that these plots match the MorletFilter.png file given in the near folder.

# In[ ]:


## Test the morlet filter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mor_filter = MorletFilterOp(1, 1, 1)  # Just set input, output, num_units to any value
    mor_filter.parameters["s"] = 0.60
    mor_filter.parameters["w"] = 0.30
    weights = mor_filter.get_filter_default_xvals(13)
    weights = weights.detach().numpy()
    plt.figure()
    plt.plot(weights, color="orange")
    plt.show()

    asym_mor_filter = AsymMorletFilterOp(1, 1, 1)  # Just set input, output, num_units to any value
    asym_mor_filter.parameters["s2"] = 0.75
    asym_mor_filter.parameters["w2"] = 0.25
    weights = asym_mor_filter.get_filter_default_xvals(13)
    weights = weights.detach().numpy()
    plt.figure()
    plt.plot(weights, color="blue")
    plt.show()

    asym_mor_filter = AsymMorletFilterOp(1, 1, 1)  # Just set input, output, num_units to any value
    asym_mor_filter.parameters["s1"] = 0.75
    asym_mor_filter.parameters["w1"] = 0.25
    weights = asym_mor_filter.get_filter_default_xvals(13)
    weights = weights.detach().numpy()
    plt.figure()
    plt.plot(weights, color="red")
    plt.show()


# ## Starting DSL Dictionary

# In[ ]:


DSL_DICT = {('list', 'list') : [MapFunction, SimpleITE],
                        ('list', 'atom') : [OrFunction, AndFunction, RunningAverageWindow5Function],
('atom', 'atom') : [SimpleITE, CALMS21ResAngleHeadBodySelection, \
                    CALMS21SpeedSelection, CALMS21TangentialVelocitySelection, \
                    CALMS21AccelerationSelection, CALMS21RelAngleSocialSelection, \
                    CALMS21AxisRatioSelection, CALMS21OverlapBboxesSelection,
                    CALMS21MinResNoseKeypointDistSelection]}

# If not updated, the default cost is based on the number of neural modules 
# in the function
CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}


# ## DSL with Morlet Filter
# 
# Create a new DSL called DSL_DICT_MOR which replaces the RunningAverage with MorletFilterOp in the original DSL_DICT.

# In[ ]:


DSL_DICT_MOR = {('list', 'list') : [MapFunction, SimpleITE],
                        ('list', 'atom') : [OrFunction, AndFunction,
                                            MorletFilterOp],
('atom', 'atom') : [SimpleITE, CALMS21ResAngleHeadBodySelection, \
                    CALMS21SpeedSelection, CALMS21TangentialVelocitySelection, \
                    CALMS21AccelerationSelection, CALMS21RelAngleSocialSelection, \
                    CALMS21AxisRatioSelection, CALMS21OverlapBboxesSelection,
                    CALMS21MinResNoseKeypointDistSelection]}


# # DSL with Neural Module
# 
# Create a new DSL called DSL_DICT_NEUROSYM which replaces feature selects with learning a neural module over all features.

# In[ ]:


DSL_DICT_NEUROSYM = {('list', 'list') : [MapFunction, SimpleITE],
                        ('list', 'atom') : [OrFunction, AndFunction, MorletFilterOp],
('atom', 'atom') : [SimpleITE, CALMS21ResAngleHeadBodySelection, \
                    CALMS21SpeedSelection, CALMS21TangentialVelocitySelection, \
                    CALMS21AccelerationSelection, CALMS21RelAngleSocialSelection, \
                    CALMS21AxisRatioSelection, CALMS21OverlapBboxesSelection,
                    CALMS21MinResNoseKeypointDistSelection, NeuralFeatureSelectionFunction]}


# # DSL with Asymmetric Morlet Filter
# 
# Create a new DSL called DSL_DICT_ASYM_MOR which uses your new implementation of the asymmetric Morlet Filter.

# In[ ]:


DSL_DICT_ASYM_MOR = {('list', 'list') : [MapFunction, SimpleITE],
                        ('list', 'atom') : [OrFunction, AndFunction,
                                            AsymMorletFilterOp],
('atom', 'atom') : [SimpleITE, CALMS21ResAngleHeadBodySelection, \
                    CALMS21SpeedSelection, CALMS21TangentialVelocitySelection, \
                    CALMS21AccelerationSelection, CALMS21RelAngleSocialSelection, \
                    CALMS21AxisRatioSelection, CALMS21OverlapBboxesSelection,
                    CALMS21MinResNoseKeypointDistSelection]}

