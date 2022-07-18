#!/usr/bin/env python
# coding: utf-8

# ## NEAR Code

# In[ ]:


import argparse
import os
import logging
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time

from dsl import StartFunction, ITE, HeuristicNeuralFunction, LibraryFunction,                 ListToListModule, ListToAtomModule, OrFunction, AndFunction
from collections.abc import Iterable


# ## Utilities

# ### Data Processing

# In[ ]:


def flatten_batch(batch):
    if not isinstance(batch[0], Iterable) or len(batch[0]) == 1:
        return batch
    new_batch = []
    for traj_list in batch:
        new_batch.extend(traj_list)
    return new_batch

def flatten_tensor(batch_out):
    return torch.cat(batch_out)

def pad_minibatch(minibatch, num_features=-1, pad_token=-1, return_max=False):
    #minibatch to be of dimension [num_sequences, num_features, len of each sequence (variable)]
    #adapted from Will Falcon's blog post
    batch_lengths = [len(sequence) for sequence in minibatch]
    batch_size = len(minibatch)
    longest_seq = max(batch_lengths)
    padded_minibatch = torch.ones((batch_size, longest_seq, num_features)) * pad_token
    for i, seq_len in enumerate(batch_lengths):
        seq = minibatch[i]
        if num_features == 1:
            seq = seq.unsqueeze(1)
        padded_minibatch[i, 0:seq_len] = seq[:seq_len]
    if return_max:
        return padded_minibatch, batch_lengths, longest_seq
    else:
        return padded_minibatch, batch_lengths

def unpad_minibatch(minibatch, lengths, listtoatom=False):
    new_minibatch = []
    for idx, length in enumerate(lengths):
        if listtoatom:
            new_minibatch.append(minibatch[idx][length-1])
        else:
            new_minibatch.append(minibatch[idx][:length])
    return new_minibatch

def dataset_tolists(trajs, labels):
    assert len(trajs) == len(labels)

    dataset = []
    for k, traj in enumerate(trajs):
        traj_list = []
        for t in range(len(traj)):
            traj_list.append(traj[t])

        label = torch.tensor(labels[k]).long()
        dataset.append([traj_list, label])

    return dataset

def normalize_data(train_data, valid_data, test_data):
    """Normalize features wrt. mean and std of training data."""
    _, seq_len, input_dim = train_data.shape
    train_data_reshape = np.reshape(train_data, (-1, input_dim))
    test_data_reshape = np.reshape(test_data, (-1, input_dim))
    features_mean = np.mean(train_data_reshape, axis=0)
    features_std = np.std(train_data_reshape, axis=0)
    train_data_reshape = (train_data_reshape - features_mean) / features_std
    test_data_reshape = (test_data_reshape - features_mean) / features_std
    train_data = np.reshape(train_data_reshape, (-1, seq_len, input_dim))
    test_data = np.reshape(test_data_reshape, (-1, seq_len, input_dim))
    if valid_data is not None:
        valid_data_reshape = np.reshape(valid_data, (-1, input_dim))
        valid_data_reshape = (valid_data_reshape - features_mean) / features_std
        valid_data = np.reshape(valid_data_reshape, (-1, seq_len, input_dim))
    return train_data, valid_data, test_data

def create_minibatches(all_items, batch_size):
    num_items = len(all_items)
    batches = []
    def create_single_minibatch(idxseq):
        curr_batch = []
        for idx in idxseq:
            curr_batch.append((all_items[idx]))
        return curr_batch
    item_idxs = list(range(num_items))
    while len(item_idxs) > 0:
        if len(item_idxs) <= batch_size:
            batch = create_single_minibatch(item_idxs)
            batches.append(batch)
            item_idxs = []
        else:
            # get batch indices
            batchidxs = []
            while len(batchidxs) < batch_size:
                rando = random.randrange(len(item_idxs))
                index = item_idxs.pop(rando)
                batchidxs.append(index)
            batch = create_single_minibatch(batchidxs)
            batches.append(batch)
    return batches

def prepare_datasets(train_data, valid_data, test_data, train_labels, valid_labels, test_labels, normalize=True, train_valid_split=0.7, batch_size=32):
    if normalize:
        train_data, valid_data, test_data = normalize_data(train_data, valid_data, test_data)

    trainset = dataset_tolists(train_data, train_labels) 
    testset = dataset_tolists(test_data, test_labels)

    if valid_data is not None and valid_labels is not None:
        validset = dataset_tolists(valid_data, valid_labels)
    # Split training for validation set if validation set is not provided.
    elif train_valid_split < 1.0:
        split = int(train_valid_split*len(train_data))
        validset = trainset[split:]
        trainset = trainset[:split]
    else:
        split = int(train_valid_split)
        validset = trainset[split:]
        trainset = trainset[:split]

    # Create minibatches for training
    batched_trainset = create_minibatches(trainset, batch_size)

    return batched_trainset, validset, testset


# ### Logging

# In[ ]:


def init_logging(save_path):
    logfile = os.path.join(save_path, 'log.txt')

    # clear log file
    with open(logfile, 'w'):
        pass
    # remove previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=logfile, level=logging.INFO)

def log_and_print(line):
    print(line)
    logging.info(line)

def log(line):
    logging.info(line)

def print_program(program, ignore_constants=True):
    if not isinstance(program, LibraryFunction):
        return program.name
    else:
        collected_names = []
        for submodule, functionclass in program.submodules.items():
            collected_names.append(print_program(functionclass, ignore_constants=ignore_constants))
        # log_and_print(program.has_params)
        if program.has_params:
            parameters = "params: {}".format(program.parameters.values())
            if not ignore_constants:
                collected_names.append(parameters)
        joined_names = ', '.join(collected_names)
        return program.name + "(" + joined_names + ")"

def print_program_dict(prog_dict):
    log_and_print(print_program(prog_dict["program"], ignore_constants=True))
    log_and_print("struct_cost {:.4f} | score {:.4f} | path_cost {:.4f} | time {:.4f}".format(
        prog_dict["struct_cost"], prog_dict["score"], prog_dict["path_cost"], prog_dict["time"]))


# ### Training

# In[ ]:


def init_optimizer(program, optimizer, lr):
    queue = [program]
    all_params = []
    while len(queue) != 0:
        current_function = queue.pop()
        if issubclass(type(current_function), HeuristicNeuralFunction):
            current_function.init_model()
            all_params.append({'params' : current_function.model.parameters(),'lr' : lr})
        elif current_function.has_params:
            if current_function.parameters is None:
                current_function.init_params()
            all_params.append({'params': list(current_function.parameters.values()), 'lr': lr})
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
        elif 'custom_nn' in current_function.__dict__ and current_function.custom_nn:
            all_params.append({'params' : current_function.model.parameters(),'lr' : lr})
        else:
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
    curr_optim = optimizer(all_params, lr)
    return curr_optim

def process_batch(program, batch, output_type, output_size, device='cpu'):
    batch_input = [torch.tensor(traj) for traj in batch]
    batch_padded, batch_lens = pad_minibatch(batch_input, num_features=batch_input[0].size(1))
    batch_padded = batch_padded.to(device)
    out_padded = program.execute_on_batch(batch_padded, batch_lens)
    if output_type == "list":
        out_unpadded = unpad_minibatch(out_padded, batch_lens, listtoatom=(program.output_type=='atom'))
    else:
        out_unpadded = out_padded
    if output_size == 1 or output_type == "list":
        if not isinstance(out_unpadded, list):
            return out_unpadded.squeeze(dim=1)
        return flatten_tensor(out_unpadded).squeeze()
    else:
        if isinstance(out_unpadded, list):
            out_unpadded = torch.cat(out_unpadded, dim=0).to(device)          
        return out_unpadded

def execute_and_train(program, validset, trainset, train_config, output_type, output_size, 
    neural=False, device='cpu', use_valid_score=False, print_every=60):

    lr = train_config['lr']
    neural_epochs = train_config['neural_epochs']
    symbolic_epochs = train_config['symbolic_epochs']
    optimizer = train_config['optimizer']
    lossfxn = train_config['lossfxn']
    evalfxn = train_config['evalfxn']
    num_labels = train_config['num_labels']

    num_epochs = neural_epochs if neural else symbolic_epochs

    # initialize optimizer
    curr_optim = init_optimizer(program, optimizer, lr)

    # prepare validation set
    validation_input, validation_output = map(list, zip(*validset))
    validation_true_vals = torch.tensor(flatten_batch(validation_output)).float().to(device)
    # TODO a little hacky, but easiest solution for now
    if isinstance(lossfxn, nn.CrossEntropyLoss):
        validation_true_vals = validation_true_vals.long()

    best_program = None
    best_metric = float('inf')
    best_additional_params = {}

    for epoch in range(1, num_epochs+1):
        for batchidx in range(len(trainset)):
            batch_input, batch_output = map(list, zip(*trainset[batchidx]))
            true_vals = torch.tensor(flatten_batch(batch_output)).float().to(device)
            predicted_vals = process_batch(program, batch_input, output_type, output_size, device)
            # TODO a little hacky, but easiest solution for now
            if isinstance(lossfxn, nn.CrossEntropyLoss):
                true_vals = true_vals.long()
            #print(predicted_vals.shape, true_vals.shape)
            loss = lossfxn(predicted_vals, true_vals)
            curr_optim.zero_grad()
            loss.backward()
            curr_optim.step()

            # if batchidx % print_every == 0 or batchidx == 0:
            #     log_and_print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

        # check score on validation set
        with torch.no_grad():
            predicted_vals = process_batch(program, validation_input, output_type, output_size, device)
            metric, additional_params = evalfxn(predicted_vals, validation_true_vals, num_labels=num_labels)

        if use_valid_score:
            if metric < best_metric:
                best_program = copy.deepcopy(program)
                best_metric = metric
                best_additional_params = additional_params
        else:
            best_program = copy.deepcopy(program)
            best_metric = metric
            best_additional_params = additional_params

    # select model with best validation score
    program = copy.deepcopy(best_program)
    log("Validation score is: {:.4f}".format(best_metric))
    log("Average f1-score is: {:.4f}".format(1 - best_metric))
    log("Hamming accuracy is: {:.4f}".format(best_additional_params['hamming_accuracy']))
    
    return best_metric


# ### Evaluation

# In[ ]:


from sklearn.metrics import hamming_loss, f1_score

def compute_average_f1_score(predicted, truth, num_labels):
    assert isinstance(predicted, torch.Tensor)
    assert isinstance(truth, torch.Tensor)

    if num_labels > 1:
        weighted_avg_f1 = f1_score(truth, predicted, average='weighted')
        unweighted_avg_f1 = f1_score(truth, predicted, average='macro')
        all_f1 = f1_score(truth, predicted, average=None)
        return weighted_avg_f1, unweighted_avg_f1, all_f1
    else:
        avg_f1 = f1_score(truth, predicted, average='binary')
        all_f1 = f1_score(truth, predicted, average=None)
        return avg_f1, all_f1

def label_correctness(predictions, truths, num_labels=1):
    #counts up hamming distance and true accuracy
    # assert predictions.size(-1) == num_labels

    additional_scores = {}
    if len(predictions.size()) == 1:
        predictions = torch.sigmoid(predictions) > 0.5
    else:
        assert len(predictions.size()) == 2
        predictions = torch.max(predictions, dim=-1)[1]

    additional_scores['hamming_accuracy'] = 1 - hamming_loss(truths.squeeze().cpu(), predictions.squeeze().cpu())
    if num_labels > 1:
        w_avg_f1, additional_scores['unweighted_f1'], additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores
    else:
        w_avg_f1, additional_scores['all_f1s'] = compute_average_f1_score(truths.squeeze().cpu(), predictions.squeeze().cpu(), num_labels)
        return 1 - w_avg_f1, additional_scores

def test_set_eval(program, testset, output_type, output_size, num_labels, device='cpu', verbose=False):
    log_and_print("\n")
    log_and_print("Evaluating program {} on TEST SET".format(print_program(program, ignore_constants=(not verbose))))
    with torch.no_grad():
        test_input, test_output = map(list, zip(*testset))
        true_vals = torch.tensor(flatten_batch(test_output)).to(device)
        predicted_vals = process_batch(program, test_input, output_type, output_size, device)
        metric, additional_params = label_correctness(predicted_vals, true_vals, num_labels=num_labels)
    log_and_print("F1 score achieved is {:.4f}".format(1 - metric))
    log_and_print("Additional performance parameters: {}\n".format(additional_params))


# ## Program Graph

# In[ ]:


class ProgramNode(object):

    def __init__(self, program, score, parent, depth, cost, order):
        self.score = score
        self.program = program
        self.children = []
        self.parent = parent
        self.depth = depth
        self.cost = cost
        self.order = order


class ProgramGraph(object):

    def __init__(self, dsl_dict, edge_cost_dict, input_type, output_type, input_size, output_size,
        max_num_units, min_num_units, max_num_children, max_depth, penalty, ite_beta=1.0, device="cpu"):
        self.dsl_dict = dsl_dict
        self.edge_cost_dict = edge_cost_dict
        self.input_type = input_type
        self.output_type = output_type
        self.input_size = input_size
        self.output_size = output_size
        self.max_num_units = max_num_units
        self.min_num_units = min_num_units
        self.max_num_children = max_num_children
        self.max_depth = max_depth
        self.penalty = penalty
        self.ite_beta = ite_beta
        self.device = device
        start = StartFunction(input_type=input_type, output_type=output_type, 
            input_size=input_size, output_size=output_size, num_units=max_num_units, device=self.device)
        self.root_node = ProgramNode(start, 0, None, 0, 0, 0)

    def construct_candidates(self, input_type, output_type, input_size, output_size, num_units):
        candidates = []
        replacement_candidates = self.dsl_dict[(input_type, output_type)]
        for functionclass in replacement_candidates:
            if issubclass(functionclass, ITE):
                candidate = functionclass(input_type, output_type, input_size, output_size, num_units, beta=self.ite_beta, device=self.device)
            elif issubclass(functionclass, OrFunction) or issubclass(functionclass, AndFunction):
                candidate = functionclass(input_type, output_type, input_size, output_size, num_units, device=self.device)
            else:
                candidate = functionclass(input_size, output_size, num_units, device=self.device)
            candidates.append(candidate)
        return candidates

    def is_fully_symbolic(self, candidate_program):
        queue = [(candidate_program.submodules['program'])]
        while len(queue) != 0:
            current_function = queue.pop()
            if issubclass(type(current_function), HeuristicNeuralFunction):
                return False
            else:
                for submodule in current_function.submodules:
                    queue.append(current_function.submodules[submodule])
        return True

    def compute_edge_cost(self, expandion_candidate):
        edge_cost = 0
        functionclass = type(expandion_candidate)
        typesig = expandion_candidate.get_typesignature()

        if functionclass in self.edge_cost_dict[typesig]:
            edge_cost = self.edge_cost_dict[typesig][functionclass]
        else:
            # Otherwise, the edge cost scales with the number of HeuristicNeuralFunction
            for submodule, fxnclass in expandion_candidate.submodules.items():
                if isinstance(fxnclass, HeuristicNeuralFunction):
                    edge_cost += 1

        return edge_cost*self.penalty

    def compute_program_cost(self, candidate_program):
        queue = [candidate_program.submodules['program']]
        total_cost = 0
        depth = 0
        edge_cost = 0
        while len(queue) != 0:
            depth += 1
            current_function = queue.pop()
            current_type = type(current_function)
            current_type_sig = current_function.get_typesignature()
            if current_type in self.edge_cost_dict[current_type_sig]:
                edge_cost = self.edge_cost_dict[current_type_sig][current_type]
            else:
                edge_cost = 0
                # Otherwise, the edge cost scales with the number of neural modules
                for submodule, fxnclass in current_function.submodules.items():
                    edge_cost += 1
            total_cost += edge_cost
            for submodule, functionclass in current_function.submodules.items():
                queue.append(functionclass)
        return total_cost * self.penalty, depth

    def min_depth2go(self, candidate_program):
        depth2go = 0
        queue = [(candidate_program.submodules['program'])]
        while len(queue) != 0:
            current_function = queue.pop()
            if issubclass(type(current_function), HeuristicNeuralFunction):
                depth2go += 1
                # in current DSL, ListToList/ListToAtom both need at least 2 depth to be fully symbolic
                if issubclass(type(current_function), ListToListModule):
                    depth2go += 1
                elif issubclass(type(current_function), ListToAtomModule):
                    depth2go += 1
            else:
                for submodule in current_function.submodules:
                    queue.append(current_function.submodules[submodule])
        return depth2go

    def num_units_at_depth(self, depth):
        num_units = max(int(self.max_num_units*(0.5**(depth-1))), self.min_num_units)
        return num_units

    def get_all_children(self, current_node, in_enumeration=False):
        all_children = []
        child_depth = current_node.depth + 1
        child_num_units = self.num_units_at_depth(child_depth)
        queue = [current_node.program]
        while len(queue) != 0:
            current = queue.pop()
            for submod, functionclass in current.submodules.items():
                if issubclass(type(functionclass), HeuristicNeuralFunction):
                    replacement_candidates = self.construct_candidates(functionclass.input_type,
                                                                   functionclass.output_type,
                                                                   functionclass.input_size,
                                                                   functionclass.output_size,
                                                                   child_num_units)
                    orig_fclass = copy.deepcopy(current.submodules[submod])
                    for child_candidate in replacement_candidates:
                        # replace the neural function with a candidate
                        current.submodules[submod] = child_candidate
                        # create the correct child node
                        child_node = copy.deepcopy(current_node)
                        child_node.depth = child_depth
                        # check if child program can be completed within max_depth
                        if child_node.depth + self.min_depth2go(child_node.program) > self.max_depth:
                            continue
                        # if yes, compute costs and add to list of children
                        child_node.cost = current_node.cost + self.compute_edge_cost(child_candidate)
                        all_children.append(child_node)
                        if len(all_children) >= self.max_num_children and not in_enumeration:
                            return all_children
                    # once we've copied it, set current back to the original current
                    current.submodules[submod] = orig_fclass
                    if not in_enumeration:
                        return all_children
                else:
                    #add submodules
                    queue.append(functionclass)
        return all_children


# ## Search Algorithm

# In[ ]:


class ProgramLearningAlgorithm(object):
    
    def __init__(self, **kwargs):
        pass

    def run(self, **kwargs):
        raise NotImplementedError


class ProgramNodeFrontier(object):
    
    def __init__(self, capacity=float('inf')):
        self.capacity = capacity
        self.prioqueue = []

    def __len__(self):
        return len(self.prioqueue)

    def add(self, item):
        assert len(item) == 3
        assert isinstance(item[2], ProgramNode)
        self.prioqueue.append(item)
        if len(self.prioqueue) > self.capacity:
            # self.sort(tup_idx=0)
            popped_f_score, _, popped = self.pop(-1)
            log_and_print("POP {} with fscore {:.4f}".format(print_program(popped.program, ignore_constants=True), popped_f_score))

    def peek(self, idx=0):
        if len(self.prioqueue) == 0:
            return None
        return self.prioqueue[idx]

    def pop(self, idx, sort_fscores=True):
        """Pops the first item off the queue."""
        if len(self.prioqueue) == 0:
            return None
        if sort_fscores:
            self.sort(tup_idx=0)
        return self.prioqueue.pop(idx)

    def sort(self, tup_idx=0):
        self.prioqueue.sort(key=lambda x: x[tup_idx])


# ### ASTAR

# In[ ]:


class ASTAR_NEAR(ProgramLearningAlgorithm):

    def __init__(self, frontier_capacity=float('inf')):
        self.frontier_capacity = frontier_capacity

    def run(self, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training root program ...")
        current = copy.deepcopy(graph.root_node)
        initial_score = execute_and_train(current.program, validset, trainset, train_config, 
            graph.output_type, graph.output_size, neural=True, device=device)
        log_and_print("Initial training complete. F1-Score from program is {:.4f} \n".format(1 - initial_score))
        
        order = 0
        frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
        frontier.add((float('inf'), order, current))
        num_children_trained = 0
        start_time = time.time()

        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []

        while len(frontier) != 0:
            current_f_score, _, current = frontier.pop(0)
            log_and_print("CURRENT program has depth {}, fscore {:.4f}: {}".format(
                current.depth, current_f_score, print_program(current.program, ignore_constants=(not verbose))))
            log("Creating children for current node/program")
            children_nodes = graph.get_all_children(current)
            # prune if more than self.max_num_children
            if len(children_nodes) > graph.max_num_children:
                children_nodes = random.sample(children_nodes, k=graph.max_num_children)  # sample without replacement
            log("{} total children to train for current node".format(len(children_nodes)))

            for child_node in children_nodes:
                child_start_time = time.time()
                log_and_print("Training child program: {}".format(print_program(child_node.program, ignore_constants=(not verbose))))
                is_neural = not graph.is_fully_symbolic(child_node.program)
                child_node.score = execute_and_train(child_node.program, validset, trainset, train_config, 
                    graph.output_type, graph.output_size, neural=is_neural, device=device)
                log("Time to train child {:.3f}".format(time.time() - child_start_time))
                num_children_trained += 1
                log("{} total children trained".format(num_children_trained))
                child_node.parent = current
                child_node.children = []
                order -= 1
                child_node.order = order  # insert order of exploration as tiebreaker for equivalent f-scores
                current.children.append(child_node)

                # computing path costs (f_scores)
                child_f_score = child_node.cost + child_node.score # cost + heuristic
                log_and_print("DEBUG: f-score {}".format(child_f_score))

                if not is_neural and child_f_score < best_total_cost:
                    best_program = copy.deepcopy(child_node.program)
                    best_total_cost = child_f_score
                    best_programs_list.append({
                            "program" : best_program,
                            "struct_cost" : child_node.cost, 
                            "score" : child_node.score,
                            "path_cost" : child_f_score,
                            "time" : time.time()-start_time
                        })
                    log_and_print("New BEST program found:")
                    print_program_dict(best_programs_list[-1])

                if is_neural:
                    assert child_node.depth < graph.max_depth
                    child_tuple = (child_f_score, order, child_node)
                    frontier.add(child_tuple)

            # clean up frontier
            frontier.sort(tup_idx=0)
            while len(frontier) > 0 and frontier.peek(-1)[0] > best_total_cost:
                frontier.pop(-1)
            log_and_print("Frontier length is: {}".format(len(frontier)))
            log_and_print("Total time elapsed is {:.3f}\n".format(time.time()-start_time))

        if best_program is None:
            log_and_print("ERROR: no program found")

        return best_programs_list


# ### IDDFS

# In[ ]:


class IDDFS_NEAR(ProgramLearningAlgorithm):

    def __init__(self, frontier_capacity=float('inf'), initial_depth=1, performance_multiplier=1.0, depth_bias=1.0, exponent_bias=False):
        self.frontier_capacity = frontier_capacity
        self.initial_depth = initial_depth
        self.performance_multiplier = performance_multiplier # < 1.0 prunes more aggressively
        self.depth_bias =depth_bias # < 1.0 prunes more aggressively
        self.exponent_bias = exponent_bias # flag to determine it depth_bias should be exponentiated or not

    def run(self, graph, trainset, validset, train_config, device, verbose=False, trainset_neural=None):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training root program ...")
        current = copy.deepcopy(graph.root_node)
        dataset = trainset_neural if trainset_neural is not None else trainset
        initial_score = execute_and_train(current.program, validset, dataset, train_config,
            graph.output_type, graph.output_size, neural=True, device=device)
        log_and_print("Initial training complete. Score from program is {:.4f} \n".format(1 - initial_score))
        
        # Branch-and-bound search with iterative deepening
        current_depth = self.initial_depth
        current_f_score = float('inf')
        order = 0
        frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
        next_frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
        num_children_trained = 0
        start_time = time.time()

        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []

        log("Starting iterative deepening with depth {}\n".format(current_depth))
        while current_depth <= graph.max_depth:
            log_and_print("CURRENT program has depth {}, fscore {:.4f}: {}".format(
                current.depth, current_f_score, print_program(current.program, ignore_constants=(not verbose))))
            log("Creating children for current node/program")
            log("Total time elapsed is {:.3f}".format(time.time()-start_time))
            children_nodes = graph.get_all_children(current)

            # prune if more than self.max_num_children
            if len(children_nodes) > graph.max_num_children:
                log("Sampling {}/{} children".format(graph.max_num_children, len(children_nodes)))
                children_nodes = random.sample(children_nodes, k=graph.max_num_children) # sample without replacement
            log("{} total children to train for current node".format(len(children_nodes)))
            
            child_tuples = []
            for child_node in children_nodes:
                child_start_time = time.time()
                log("Training child program: {}".format(print_program(child_node.program, ignore_constants=(not verbose))))
                is_neural = not graph.is_fully_symbolic(child_node.program)
                dataset = trainset_neural if (is_neural and trainset_neural is not None) else trainset
                child_node.score = execute_and_train(child_node.program, validset, dataset, train_config,
                    graph.output_type, graph.output_size, neural=is_neural, device=device)
                log("Time to train child {:.3f}".format(time.time()-child_start_time))
                num_children_trained += 1
                log("{} total children trained".format(num_children_trained))
                child_node.parent = current
                child_node.children = []
                order -= 1
                child_node.order = order # insert order of exploration as tiebreaker for equivalent f-scores

                # computing path costs (f_scores)
                child_f_score = child_node.cost + child_node.score # cost + heuristic
                log("DEBUG: f-score {}".format(child_f_score))
                current.children.append(child_node)
                child_tuples.append((child_f_score, order, child_node))

                if not is_neural and child_f_score < best_total_cost:
                    best_program = copy.deepcopy(child_node.program)
                    best_total_cost = child_f_score
                    best_programs_list.append({
                            "program" : best_program,
                            "struct_cost" : child_node.cost, 
                            "score" : child_node.score,
                            "path_cost" : child_f_score,
                            "time" : time.time()-start_time
                        })
                    log_and_print("New BEST program found:")
                    print_program_dict(best_programs_list[-1])

            # find next current among children, from best to worst
            nextfound = False
            child_tuples.sort(key=lambda x: x[0])
            for child_tuple in child_tuples:
                child = child_tuple[2]
                if graph.is_fully_symbolic(child.program):
                    continue # don't want to expand symbolic programs (no children)
                elif child.depth >= current_depth:
                    next_frontier.add(child_tuple)
                else:
                    if not nextfound:
                        nextfound = True # first child program that's not symbolic and within current_depth
                        current_f_score, current_order, current = child_tuple
                        log_and_print("Found program among children: {} with f_score {}".format(
                            print_program(current.program, ignore_constants=(not verbose)), current_f_score))
                    else:
                        frontier.add(child_tuple) # put the rest onto frontier

            # find next node in frontier
            if not nextfound:
                frontier.sort(tup_idx=1) # DFS order
                log_and_print("Frontier length is: {}".format(len(frontier)))
                original_depth = current.depth
                while len(frontier) > 0 and not nextfound:
                    current_f_score, current_order, current = frontier.pop(0, sort_fscores=False) # DFS order
                    if current_f_score < self.bound_modify(best_total_cost, original_depth, current.depth):
                        nextfound = True
                        log_and_print("Found program in frontier: {} with f_score {}".format(
                            print_program(current.program, ignore_constants=(not verbose)), current_f_score))
                    else:
                        log_and_print("PRUNE from frontier: {} with f_score {}".format(
                            print_program(current.program, ignore_constants=(not verbose)), current_f_score))
                log_and_print("Frontier length is now {}".format(len(frontier)))

            # frontier is empty, go to next stage of iterative deepening
            if not nextfound:
                assert len(frontier) == 0
                log_and_print("Empty frontier, moving to next depth level")
                log_and_print("DEBUG: time since start is {:.3f}\n".format(time.time() - start_time))

                current_depth += 1

                if current_depth > graph.max_depth:
                    log_and_print("Max depth {} reached. Exiting.\n".format(graph.max_depth))
                    break
                elif len(next_frontier) == 0:
                    log_and_print("Next frontier is empty. Exiting.\n")
                    break
                else:
                    log_and_print("Starting iterative deepening with depth {}\n".format(current_depth))
                    frontier = copy.deepcopy(next_frontier)
                    next_frontier = ProgramNodeFrontier(capacity=self.frontier_capacity)
                    current_f_score, current_order, current = frontier.pop(0)

        if best_program is None:
            log_and_print("ERROR: no program found")

        return best_programs_list

    def bound_modify(self, upperbound, current_depth, node_depth):
        if not self.exponent_bias:
            depth_multiplier = self.performance_multiplier * (self.depth_bias**(current_depth-node_depth))
        else:
            depth_multiplier = self.performance_multiplier ** (self.depth_bias**(current_depth-node_depth))
        return upperbound * depth_multiplier


# ### Enumeration

# In[ ]:


class ENUMERATION(ProgramLearningAlgorithm):

    def __init__(self, min_enum_depth=1, max_num_programs=100):
        self.min_enum_depth = min_enum_depth
        self.max_num_programs = max_num_programs

    def run(self, graph, trainset, validset, train_config, device, verbose=False, neur_trainset=None,
            neur_validset=None):
        assert isinstance(graph, ProgramGraph)

        symbolic_programs = []
        enum_depth = self.min_enum_depth
        while len(symbolic_programs) < self.max_num_programs and enum_depth <= graph.max_depth:
            print("DEBUG: starting enumerative synthesis with depth {}".format(enum_depth))
            symbolic_programs = self.enumerate2depth(graph, enum_depth)
            print("DEBUG: {} programs found".format(len(symbolic_programs)))
            enum_depth += 1

        log_and_print("Symbolic Synthesis: generated {}/{} symbolic programs from candidate program.".format(
            len(symbolic_programs), self.max_num_programs))
        
        total_eval = min(self.max_num_programs, len(symbolic_programs))
        symbolic_programs.sort(key=lambda x: x["struct_cost"])
        symbolic_programs = symbolic_programs[:total_eval]

        best_total_cost = float('inf')
        start_time = time.time()
        num_programs_trained = 1
        top_programs_list = []
        for prog_dict in symbolic_programs:
            child_start_time = time.time()
            candidate = prog_dict["program"]
            log_and_print("Training candidate program ({}/{}) {}".format(
                num_programs_trained, total_eval, print_program(candidate, ignore_constants=(not verbose))))
            num_programs_trained += 1
            score = execute_and_train(candidate, validset, trainset, train_config, 
                graph.output_type, graph.output_size, neural=False, device=device)
            total_cost = score + prog_dict["struct_cost"]
            log_and_print("Structural cost is {} with structural penalty {}".format(prog_dict["struct_cost"], graph.penalty))
            log_and_print("Time to train child {:.3f}".format(time.time()-child_start_time))
            log_and_print("Total time elapsed is: {:.3f}".format(time.time()-start_time))

            prog_dict["score"] = score
            prog_dict["path_cost"] = total_cost
            prog_dict["time"] = time.time() - start_time
            top_programs_list.append(prog_dict)
            top_programs_list.sort(key=lambda i: i['path_cost'], reverse=True)
            top_programs_list = top_programs_list#[-self.num_re_eval:]
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                log_and_print("New tentative BEST program found:")
                print_program_dict(top_programs_list[-1])

        for program_dict in top_programs_list:
            # retrain each program
            candidate = program_dict["program"]
            program_dict["program"] = copy.deepcopy(candidate)
        top_programs_list.sort(key=lambda i: i['path_cost'], reverse=True)
        log_and_print("Total time elapsed is {:.3f}".format(time.time() - start_time))
        return top_programs_list


    def enumerate2depth(self, graph, enumeration_depth):
        max_depth_copy = graph.max_depth
        graph.max_depth = enumeration_depth
        all_programs = []
        enumerated = {}
        root = copy.deepcopy(graph.root_node)
        
        def enumerate_helper(program_node):
            program_name = print_program(program_node.program, ignore_constants=True)
            assert not enumerated.get(program_name)
            enumerated[program_name] = True
            if graph.is_fully_symbolic(program_node.program):
                all_programs.append({
                        "program" : copy.deepcopy(program_node.program),
                        "struct_cost" : program_node.cost,
                        "depth" : program_node.depth
                    })
            elif program_node.depth < enumeration_depth:
                all_children = graph.get_all_children(program_node, in_enumeration=True)
                for childnode in all_children:
                    if not enumerated.get(print_program(childnode.program, ignore_constants=True)):
                        enumerate_helper(childnode)
        
        enumerate_helper(root)
        graph.max_depth = max_depth_copy

        return all_programs

