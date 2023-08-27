#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
some functions are from AllSet.
https://arxiv.org/abs/2106.13264
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.typing import Adj,  OptTensor, SparseTensor





def get_activation(act, inplace=False):
  """

  Parameters
  ----------
  act
      Name of the activation
  inplace
      Whether to perform inplace activation

  Returns
  -------
  activation_layer
      The activation
  """
  if act is None:
    return lambda x: x

  if isinstance(act, str):
    if act == 'leaky':
      # TODO(sxjscience) Add regex matching here to parse `leaky(0.1)`
      return nn.LeakyReLU(0.1, inplace=inplace)
    if act == 'identity':
      return nn.Identity()
    if act == 'elu':
      return nn.ELU(inplace=inplace)
    if act == 'gelu':
      return nn.GELU()
    if act == 'relu':
      return nn.ReLU()
    if act == 'sigmoid':
      return nn.Sigmoid()
    if act == 'tanh':
      return nn.Tanh()
    if act in {'softrelu', 'softplus'}:
      return nn.Softplus()
    if act == 'softsign':
      return nn.Softsign()
    raise NotImplementedError('act="{}" is not supported. '
                              'Try to include it if you can find that in '
                              'https://pytorch.org/docs/stable/nn.html'.format(act))

  return act




class PositionwiseFFN(nn.Module):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    Also, if we use gated projection. We will use
        fc1_1 * act(fc1_2(data)) to map the data
    """
    def __init__(self,  config):
        """
        Parameters
        ----------
        units
        hidden_size
        activation_dropout
        dropout
        activation
        normalization
            layer_norm or no_norm
        layer_norm_eps
        pre_norm
            Pre-layer normalization as proposed in the paper:
            "[ACL2018] The Best of Both Worlds: Combining Recent Advances in
             Neural Machine Translation"
            This will stabilize the training of Transformers.
            You may also refer to
            "[Arxiv2020] Understanding the Difficulty of Training Transformers"
        """
        super().__init__()
        self.config = config
        self.dropout_layer = nn.Dropout(self.config.hidden_dropout_prob)
        self.activation_dropout_layer = nn.Dropout(self.config.activation_dropout)
        self.ffn_1 = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.intermediate_size,
                               bias=True)
        if self.config.gated_proj:
            self.ffn_1_gate = nn.Linear(in_features=self.config.hidden_size,
                                        out_features=self.config.hidden_size,
                                        bias=True)
        self.activation = get_activation(self.config.hidden_act)
        self.ffn_2 = nn.Linear(in_features=self.config.intermediate_size, out_features=self.config.hidden_size,
                               bias=True)
        self.layer_norm =  nn.LayerNorm(eps=self.config.layer_norm_eps,
                                   normalized_shape=self.config.hidden_size)
        self.init_weights()

    def init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """

        Parameters
        ----------
        data :
            Shape (B, seq_length, C_in)

        Returns
        -------
        out :
            Shape (B, seq_length, C_out)
        """
        residual = data
        if self.config.pre_norm:
            data = self.layer_norm(data)
        if self.config.gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self.config.pre_norm:
            out = self.layer_norm(out)
        return out




# Method for initialization
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class AllSetTrans(MessagePassing):
    """
        AllSetTrans part:
        Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
        i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
        In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """

    def __init__(self, config, negative_slope=0.2, **kwargs):

        super(AllSetTrans, self).__init__(node_dim=0, **kwargs)

        self.in_channels = config.hidden_size
        self.heads = config.num_attention_heads
        self.hidden = config.hidden_size // self.heads
        self.out_channels = config.hidden_size

        self.negative_slope = negative_slope
        self.dropout = config.attention_probs_dropout_prob
        self.aggr = 'add'

        self.lin_K = Linear(self.in_channels, self.heads * self.hidden)
        self.lin_V = Linear(self.in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(1, self.heads, self.hidden))  # Seed vector
        self.rFF = PositionwiseFFN(config)


        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)


    def forward(self, x, edge_index: Adj, return_attention_weights=None):
        """
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.hidden
        alpha_r: OptTensor = None
        
        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_K = self.lin_K(x).view(-1, H, C)
        x_V = self.lin_V(x).view(-1, H, C)
        alpha_r = (x_K * self.att_r).sum(dim=-1)

        out = self.propagate(edge_index, x=x_V,
                             alpha=alpha_r, aggr=self.aggr)

        alpha = self._alpha
        self._alpha = None
        out += self.att_r  # Seed + Multihead
        # concat heads then LayerNorm.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection.
        out = self.ln1(out + F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j,
                index, ptr,):
        
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max() + 1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        if aggr is None:
            aggr = self.aggr
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)