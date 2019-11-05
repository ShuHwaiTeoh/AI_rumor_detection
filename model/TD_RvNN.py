import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.autograd
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm

################################ tree rnn class ######################################
class RvNN(nn.Module):
    """Data is represented in a tree structure.

    Every leaf and internal node has a data (provided by the input)
    and a memory or hidden state.  The hidden state is computed based
    on its own data and the hidden states of its parents.  The
    hidden state of leaves is given by a custom init function.

    The entire tree's embedding is represented by the final
    state computed at the root.

    """
    def __init__(self, word_dim=5000, hidden_dim=100, Nclass=4):
        super(RvNN, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.hidden_dim = hidden_dim
        self.Nclass = Nclass
        #GRU
        #parameter matrix for transforming input vector of current node
        self.E = self.init_matrix([hidden_dim, word_dim])
        #weight and bias of update gate vector
        self.W_z = self.init_matrix([hidden_dim, hidden_dim])
        self.U_z = self.init_matrix([hidden_dim, hidden_dim])
        self.b_z = torch.FloatTensor(self.init_vector([self.hidden_dim]))
        # weight and bias of reset gate vector
        self.W_r = self.init_matrix([hidden_dim, hidden_dim])
        self.U_r = self.init_matrix([hidden_dim, hidden_dim])
        self.b_r = torch.FloatTensor(self.init_vector([self.hidden_dim]))
        # weight and bias of FRU output
        self.W_h = self.init_matrix([hidden_dim, hidden_dim])
        self.U_h = self.init_matrix([hidden_dim, hidden_dim])
        self.b_h = torch.FloatTensor(self.init_vector([self.hidden_dim]))
        # wieght and bias of output
        self.W_out = torch.FloatTensor(self.init_matrix([self.Nclass, self.hidden_dim]))
        self.b_out = torch.FloatTensor(self.init_vector([self.Nclass]))
        self.params = [self.E.requires_grad_(), self.W_z.requires_grad_(), \
                       self.U_z.requires_grad_(), self.b_z.requires_grad_(), \
                       self.W_r.requires_grad_(), self.U_r.requires_grad_(), \
                       self.b_r.requires_grad_(), self.W_h.requires_grad_(), \
                       self.U_h.requires_grad_(), self.b_h.requires_grad_(),\
                       self.W_out.requires_grad_(), self.b_out.requires_grad_()]

    def init_matrix(self, shape):
        return torch.from_numpy(np.random.normal(scale=0.1, size=shape)).float()

    def init_vector(self, shape):
        return torch.zeros(shape, dtype=torch.float32)

    def create_recursive_unit(self, word, index, parent_h): #gated recurrent unit
        #calculate hidden state of a node
        #transformed input vector of current node: x_hat = x * E
        child_xe = torch.matmul(self.E[:,index],word)
        #update gate vector: help model determine how much past information
        #from previous time steps needs to be passed along to the future.
        hard_sigmoid = nn.Hardtanh(0, 1)
        z = hard_sigmoid(torch.matmul(self.W_z,child_xe)+torch.matmul(self.U_z,parent_h)+self.b_z)
        #reset gate vector: decide how much of the past information to forget
        r = hard_sigmoid(torch.matmul(self.W_r,child_xe)+torch.matmul(self.U_r,parent_h)+self.b_r)
        #hidden state of current node
        c = torch.tanh(torch.matmul(self.W_h,child_xe)+torch.matmul(self.U_h,(parent_h * r))+self.b_h)
        #vector which holds information for current unit and passes it down to the network
        h = (1-z)*parent_h + z*c
        return h #hidden state computed for current node

    def compute_tree(self, x_word, x_index, num_parent, tree):
        node_h = torch.zeros([len(x_word), 1, self.hidden_dim], dtype=torch.float32)
        child_hs = torch.zeros([len(x_index), 1, self.hidden_dim], dtype=torch.float32)
        for i in range(len(x_index)):
            parent_h = node_h[tree[i][0]][0].clone()
            child_h = self.create_recursive_unit(torch.from_numpy(x_word[i]), x_index[i], parent_h)
            node_h[tree[i][1]][0] = node_h[tree[i][1]][0].clone()*torch.zeros(child_h.shape)+child_h
            child_hs[i][0] = child_hs[i][0].clone()*torch.zeros(child_h.shape)+child_h
        final_state, _ = child_hs[num_parent-1:][0].max(dim=0)
        pred_y = F.softmax(torch.matmul(self.W_out, final_state) + self.b_out, dim=0)
        return pred_y

    def forwardM(self, updates):
        with torch.no_grad():
            for i in range(len(self.params)):
                self.params[i] = updates[self.params[i]]

    def zeroGrad(self):
        if self.E.grad is not None:
            self.E.grad.zero_()
        if self.W_z.grad is not None:
            self.W_z.grad.zero_()
        if self.U_z.grad is not None:
            self.U_z.grad.zero_()
        if self.b_z.grad is not None:
            self.b_z.grad.zero_()
        if self.W_r.grad is not None:
            self.W_r.grad.zero_()
        if self.U_r.grad is not None:
            self.U_r.grad.zero_()
        if self.b_r.grad is not None:
            self.b_r.grad.zero_()
        if self.W_h.grad is not None:
            self.W_h.grad.zero_()
        if self.U_h.grad is not None:
            self.U_h.grad.zero_()
        if self.b_h.grad is not None:
            self.b_h.grad.zero_()
        if self.W_out.grad is not None:
            self.W_out.grad.zero_()
        if self.b_out.grad is not None:
            self.b_out.grad.zero_()