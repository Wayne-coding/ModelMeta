import math
import torch.nn as nn
import torch
import numpy as np
from torch.nn.init import trunc_normal_

from torch.nn import CrossEntropyLoss, LayerNorm, Linear, Parameter
from network.nlp.GPT.src.utils import GPTConfig
import os

if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
    final_device = 'cuda:0'
else:
    final_device = 'cpu'




class EmbeddingLookupPyTorch(nn.Module):
    """
    The embedding lookup table for vocabulary

    Args:
        config: the config of network, which contains vocab_size, embedding_size, and seq_length

    Inputs:
        input_ids: the tokenized inputs with datatype int64

    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size, seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(EmbeddingLookupPyTorch, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.seq_length = config.seq_length
        # Create an embedding table as a parameter
        self.embedding_table = Parameter(torch.Tensor(self.vocab_size, self.embedding_size))
        # Initialize the embedding table
        trunc_normal_(self.embedding_table, mean=0, std=0.02)

    def forward(self, input_ids):
        # Ensure input_ids are long type for gather
        input_ids = input_ids.long()
        # Use torch.index_select or torch.gather to mimic MindSpore's P.Gather
        # print("self.embedding_table", self.embedding_table.device)
        # print("input_ids", input_ids.device)
        output = torch.index_select(self.embedding_table, 0, input_ids.view(-1))
        # Reshape the output to the desired shape
        output = output.view(-1, self.seq_length, self.embedding_size)
        return output, self.embedding_table


def normal_weight(shape, num_units):
    norm = np.random.normal(0.0, num_units ** -0.5, shape).astype(np.float32)
    return torch.tensor(norm).to(final_device)


class GPT_Head(nn.Module):
    """
    Head for GPT to get the logits of each token in the vocab

    Args:
        config: the config of network, which contains embedding_size and compute_dtype

    Inputs:
        state: the output of the backbone, a Tensor
        embedding_table: the embedding table of the vocabulary, a Tensor

    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(GPT_Head, self).__init__()
        self.embedding_size = config.embedding_size
        # In PyTorch, the dtype might be set globally or per operation, not stored in the class.

    def forward(self, state, embedding_table):
        state = state.reshape(-1, self.embedding_size)
        # In PyTorch, matrix multiplication with a transposed matrix is often done using torch.matmul or the @ operator.
        logits = torch.matmul(state, embedding_table.T)
        # Log softmax is a standalone functional call in PyTorch.
        logits = F.log_softmax(logits, dim=-1)
        return logits


# GPT translation
class GPTPyTorch(nn.Module):
    def __init__(self, config):
        super(GPTPyTorch, self).__init__()
        self.backbone = GPT_Model(config)
        self.head = GPT_Head(config)
        self.eos_token = 50256
        self._initialize_weights()

        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None

        self.in_shapes = {
            'backbone.get_attention_mask': [1, 1024],
            'backbone.word_embedding': [1, 1024],
            'backbone.position_embedding': [1, 1024],
            'backbone.encoder.blocks.0.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.0.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.0.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.0.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.0.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.0.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.0.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.0.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.0.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.0.output.hidden_act': [1024, 4096],
            'backbone.encoder.blocks.0.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.0.output.projection': [1024, 4096],
            'backbone.encoder.blocks.0.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.1.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.1.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.1.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.1.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.1.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.1.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.1.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.1.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.1.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.1.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.1.output.projection': [1024, 4096],
            'backbone.encoder.blocks.1.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.2.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.2.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.2.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.2.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.2.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.2.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.2.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.2.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.2.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.2.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.2.output.projection': [1024, 4096],
            'backbone.encoder.blocks.2.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.3.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.3.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.3.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.3.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.3.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.3.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.3.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.3.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.3.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.3.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.3.output.projection': [1024, 4096],
            'backbone.encoder.blocks.3.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.4.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.4.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.4.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.4.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.4.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.4.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.4.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.4.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.4.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.4.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.4.output.projection': [1024, 4096],
            'backbone.encoder.blocks.4.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.5.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.5.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.5.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.5.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.5.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.5.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.5.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.5.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.5.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.5.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.5.output.projection': [1024, 4096],
            'backbone.encoder.blocks.5.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.6.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.6.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.6.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.6.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.6.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.6.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.6.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.6.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.6.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.6.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.6.output.projection': [1024, 4096],
            'backbone.encoder.blocks.6.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.7.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.7.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.7.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.7.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.7.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.7.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.7.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.7.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.7.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.7.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.7.output.projection': [1024, 4096],
            'backbone.encoder.blocks.7.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.8.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.8.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.8.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.8.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.8.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.8.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.8.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.8.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.8.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.8.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.8.output.projection': [1024, 4096],
            'backbone.encoder.blocks.8.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.9.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.9.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.9.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.9.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.9.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.9.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.9.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.9.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.9.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.9.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.9.output.projection': [1024, 4096],
            'backbone.encoder.blocks.9.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.10.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.10.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.10.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.10.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.10.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.10.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.10.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.10.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.10.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.10.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.10.output.projection': [1024, 4096],
            'backbone.encoder.blocks.10.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.11.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.11.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.11.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.11.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.11.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.11.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.11.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.11.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.11.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.11.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.11.output.projection': [1024, 4096],
            'backbone.encoder.blocks.11.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.12.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.12.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.12.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.12.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.12.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.12.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.12.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.12.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.12.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.12.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.12.output.projection': [1024, 4096],
            'backbone.encoder.blocks.12.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.13.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.13.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.13.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.13.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.13.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.13.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.13.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.13.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.13.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.13.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.13.output.projection': [1024, 4096],
            'backbone.encoder.blocks.13.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.14.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.14.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.14.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.14.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.14.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.14.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.14.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.14.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.14.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.14.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.14.output.projection': [1024, 4096],
            'backbone.encoder.blocks.14.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.15.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.15.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.15.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.15.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.15.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.15.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.15.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.15.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.15.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.15.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.15.output.projection': [1024, 4096],
            'backbone.encoder.blocks.15.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.16.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.16.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.16.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.16.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.16.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.16.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.16.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.16.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.16.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.16.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.16.output.projection': [1024, 4096],
            'backbone.encoder.blocks.16.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.17.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.17.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.17.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.17.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.17.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.17.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.17.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.17.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.17.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.17.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.17.output.projection': [1024, 4096],
            'backbone.encoder.blocks.17.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.18.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.18.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.18.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.18.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.18.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.18.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.18.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.18.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.18.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.18.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.18.output.projection': [1024, 4096],
            'backbone.encoder.blocks.18.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.19.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.19.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.19.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.19.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.19.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.19.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.19.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.19.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.19.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.19.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.19.output.projection': [1024, 4096],
            'backbone.encoder.blocks.19.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.20.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.20.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.20.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.20.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.20.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.20.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.20.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.20.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.20.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.20.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.20.output.projection': [1024, 4096],
            'backbone.encoder.blocks.20.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.21.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.21.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.21.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.21.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.21.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.21.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.21.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.21.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.21.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.21.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.21.output.projection': [1024, 4096],
            'backbone.encoder.blocks.21.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.22.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.22.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.22.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.22.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.22.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.22.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.22.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.22.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.22.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.22.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.22.output.projection': [1024, 4096],
            'backbone.encoder.blocks.22.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.23.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.23.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.23.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.23.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.23.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.23.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.23.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.23.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.23.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.23.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.23.output.projection': [1024, 4096],
            'backbone.encoder.blocks.23.output.dropout': [1024, 1024],
            'backbone.layernorm': [1, 1024, 1024],
            'head': [1, 1024, 1024],
            'INPUT': [1, 1025],
            'OUTPUT': [1024, 50257]
        }

        self.out_shapes = {
            'backbone.get_attention_mask': [1, 1024, 1024],
            'backbone.word_embedding': [1, 1024, 1024], #delete the second output
            'backbone.position_embedding': [1, 1024, 1024],
            'backbone.encoder.blocks.0.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.0.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.0.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.0.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.0.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.0.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.0.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.0.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.0.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.0.output.hidden_act': [1024, 4096],
            'backbone.encoder.blocks.0.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.0.output.projection': [1024, 1024],
            'backbone.encoder.blocks.0.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.1.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.1.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.1.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.1.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.1.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.1.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.1.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.1.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.1.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.1.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.1.output.projection': [1024, 1024],
            'backbone.encoder.blocks.1.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.2.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.2.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.2.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.2.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.2.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.2.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.2.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.2.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.2.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.2.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.2.output.projection': [1024, 1024],
            'backbone.encoder.blocks.2.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.3.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.3.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.3.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.3.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.3.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.3.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.3.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.3.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.3.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.3.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.3.output.projection': [1024, 1024],
            'backbone.encoder.blocks.3.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.4.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.4.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.4.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.4.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.4.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.4.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.4.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.4.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.4.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.4.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.4.output.projection': [1024, 1024],
            'backbone.encoder.blocks.4.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.5.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.5.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.5.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.5.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.5.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.5.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.5.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.5.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.5.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.5.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.5.output.projection': [1024, 1024],
            'backbone.encoder.blocks.5.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.6.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.6.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.6.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.6.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.6.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.6.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.6.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.6.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.6.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.6.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.6.output.projection': [1024, 1024],
            'backbone.encoder.blocks.6.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.7.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.7.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.7.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.7.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.7.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.7.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.7.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.7.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.7.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.7.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.7.output.projection': [1024, 1024],
            'backbone.encoder.blocks.7.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.8.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.8.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.8.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.8.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.8.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.8.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.8.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.8.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.8.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.8.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.8.output.projection': [1024, 1024],
            'backbone.encoder.blocks.8.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.9.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.9.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.9.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.9.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.9.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.9.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.9.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.9.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.9.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.9.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.9.output.projection': [1024, 1024],
            'backbone.encoder.blocks.9.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.10.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.10.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.10.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.10.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.10.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.10.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.10.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.10.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.10.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.10.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.10.output.projection': [1024, 1024],
            'backbone.encoder.blocks.10.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.11.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.11.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.11.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.11.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.11.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.11.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.11.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.11.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.11.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.11.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.11.output.projection': [1024, 1024],
            'backbone.encoder.blocks.11.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.12.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.12.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.12.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.12.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.12.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.12.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.12.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.12.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.12.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.12.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.12.output.projection': [1024, 1024],
            'backbone.encoder.blocks.12.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.13.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.13.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.13.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.13.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.13.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.13.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.13.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.13.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.13.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.13.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.13.output.projection': [1024, 1024],
            'backbone.encoder.blocks.13.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.14.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.14.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.14.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.14.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.14.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.14.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.14.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.14.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.14.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.14.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.14.output.projection': [1024, 1024],
            'backbone.encoder.blocks.14.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.15.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.15.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.15.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.15.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.15.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.15.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.15.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.15.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.15.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.15.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.15.output.projection': [1024, 1024],
            'backbone.encoder.blocks.15.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.16.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.16.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.16.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.16.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.16.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.16.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.16.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.16.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.16.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.16.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.16.output.projection': [1024, 1024],
            'backbone.encoder.blocks.16.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.17.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.17.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.17.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.17.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.17.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.17.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.17.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.17.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.17.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.17.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.17.output.projection': [1024, 1024],
            'backbone.encoder.blocks.17.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.18.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.18.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.18.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.18.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.18.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.18.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.18.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.18.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.18.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.18.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.18.output.projection': [1024, 1024],
            'backbone.encoder.blocks.18.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.19.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.19.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.19.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.19.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.19.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.19.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.19.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.19.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.19.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.19.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.19.output.projection': [1024, 1024],
            'backbone.encoder.blocks.19.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.20.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.20.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.20.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.20.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.20.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.20.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.20.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.20.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.20.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.20.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.20.output.projection': [1024, 1024],
            'backbone.encoder.blocks.20.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.21.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.21.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.21.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.21.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.21.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.21.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.21.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.21.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.21.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.21.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.21.output.projection': [1024, 1024],
            'backbone.encoder.blocks.21.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.22.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.22.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.22.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.22.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.22.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.22.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.22.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.22.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.22.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.22.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.22.output.projection': [1024, 1024],
            'backbone.encoder.blocks.22.output.dropout': [1024, 1024],
            'backbone.encoder.blocks.23.layernorm1': [1024, 1024],
            'backbone.encoder.blocks.23.layernorm2': [1024, 1024],
            'backbone.encoder.blocks.23.attention.projection': [1024, 1024],
            'backbone.encoder.blocks.23.attention.dropout': [1024, 1024],
            'backbone.encoder.blocks.23.attention.prob_dropout': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.23.attention.softmax': [1, 16, 1024, 1024],
            'backbone.encoder.blocks.23.attention.dense1': [1024, 1024],
            'backbone.encoder.blocks.23.attention.dense2': [1024, 1024],
            'backbone.encoder.blocks.23.attention.dense3': [1024, 1024],
            'backbone.encoder.blocks.23.output.mapping': [1024, 4096],
            'backbone.encoder.blocks.23.output.projection': [1024, 1024],
            'backbone.encoder.blocks.23.output.dropout': [1024, 1024],
            'backbone.layernorm': [1, 1024, 1024],
            'head': [1024, 50257],
            'INPUT': [1, 1025],
            'OUTPUT': [1024, 50257]
        }

        self.orders = {
            'backbone.word_embedding': ['INPUT',
                                        ['backbone.encoder.blocks.0.layernorm1', 'backbone.encoder.blocks.0.layernorm2',
                                         'backbone.encoder.blocks.1.layernorm1', 'backbone.encoder.blocks.1.layernorm2',
                                         'backbone.encoder.blocks.2.layernorm1', 'backbone.encoder.blocks.2.layernorm2',
                                         'backbone.encoder.blocks.3.layernorm1', 'backbone.encoder.blocks.3.layernorm2',
                                         'backbone.encoder.blocks.4.layernorm1', 'backbone.encoder.blocks.4.layernorm2',
                                         'backbone.encoder.blocks.5.layernorm1', 'backbone.encoder.blocks.5.layernorm2',
                                         'backbone.encoder.blocks.6.layernorm1', 'backbone.encoder.blocks.6.layernorm2',
                                         'backbone.encoder.blocks.7.layernorm1', 'backbone.encoder.blocks.7.layernorm2',
                                         'backbone.encoder.blocks.8.layernorm1', 'backbone.encoder.blocks.8.layernorm2',
                                         'backbone.encoder.blocks.9.layernorm1', 'backbone.encoder.blocks.9.layernorm2',
                                         'backbone.encoder.blocks.10.layernorm1',
                                         'backbone.encoder.blocks.10.layernorm2',
                                         'backbone.encoder.blocks.11.layernorm1',
                                         'backbone.encoder.blocks.11.layernorm2',
                                         'backbone.encoder.blocks.12.layernorm1',
                                         'backbone.encoder.blocks.12.layernorm2',
                                         'backbone.encoder.blocks.13.layernorm1',
                                         'backbone.encoder.blocks.13.layernorm2',
                                         'backbone.encoder.blocks.14.layernorm1',
                                         'backbone.encoder.blocks.14.layernorm2',
                                         'backbone.encoder.blocks.15.layernorm1',
                                         'backbone.encoder.blocks.15.layernorm2',
                                         'backbone.encoder.blocks.16.layernorm1',
                                         'backbone.encoder.blocks.16.layernorm2',
                                         'backbone.encoder.blocks.17.layernorm1',
                                         'backbone.encoder.blocks.17.layernorm2',
                                         'backbone.encoder.blocks.18.layernorm1',
                                         'backbone.encoder.blocks.18.layernorm2',
                                         'backbone.encoder.blocks.19.layernorm1',
                                         'backbone.encoder.blocks.19.layernorm2',
                                         'backbone.encoder.blocks.20.layernorm1',
                                         'backbone.encoder.blocks.20.layernorm2',
                                         'backbone.encoder.blocks.21.layernorm1',
                                         'backbone.encoder.blocks.21.layernorm2',
                                         'backbone.encoder.blocks.22.layernorm1',
                                         'backbone.encoder.blocks.22.layernorm2',
                                         'backbone.encoder.blocks.23.layernorm1',
                                         'backbone.encoder.blocks.23.layernorm2',
                                         'backbone.layernorm', 'head']],
            'backbone.position_embedding': ['INPUT',
                                            ['backbone.encoder.blocks.0.layernorm1',
                                             'backbone.encoder.blocks.0.layernorm2',
                                             'backbone.encoder.blocks.1.layernorm1',
                                             'backbone.encoder.blocks.1.layernorm2',
                                             'backbone.encoder.blocks.2.layernorm1',
                                             'backbone.encoder.blocks.2.layernorm2',
                                             'backbone.encoder.blocks.3.layernorm1',
                                             'backbone.encoder.blocks.3.layernorm2',
                                             'backbone.encoder.blocks.4.layernorm1',
                                             'backbone.encoder.blocks.4.layernorm2',
                                             'backbone.encoder.blocks.5.layernorm1',
                                             'backbone.encoder.blocks.5.layernorm2',
                                             'backbone.encoder.blocks.6.layernorm1',
                                             'backbone.encoder.blocks.6.layernorm2',
                                             'backbone.encoder.blocks.7.layernorm1',
                                             'backbone.encoder.blocks.7.layernorm2',
                                             'backbone.encoder.blocks.8.layernorm1',
                                             'backbone.encoder.blocks.8.layernorm2',
                                             'backbone.encoder.blocks.9.layernorm1',
                                             'backbone.encoder.blocks.9.layernorm2',
                                             'backbone.encoder.blocks.10.layernorm1',
                                             'backbone.encoder.blocks.10.layernorm2',
                                             'backbone.encoder.blocks.11.layernorm1',
                                             'backbone.encoder.blocks.11.layernorm2',
                                             'backbone.encoder.blocks.12.layernorm1',
                                             'backbone.encoder.blocks.12.layernorm2',
                                             'backbone.encoder.blocks.13.layernorm1',
                                             'backbone.encoder.blocks.13.layernorm2',
                                             'backbone.encoder.blocks.14.layernorm1',
                                             'backbone.encoder.blocks.14.layernorm2',
                                             'backbone.encoder.blocks.15.layernorm1',
                                             'backbone.encoder.blocks.15.layernorm2',
                                             'backbone.encoder.blocks.16.layernorm1',
                                             'backbone.encoder.blocks.16.layernorm2',
                                             'backbone.encoder.blocks.17.layernorm1',
                                             'backbone.encoder.blocks.17.layernorm2',
                                             'backbone.encoder.blocks.18.layernorm1',
                                             'backbone.encoder.blocks.18.layernorm2',
                                             'backbone.encoder.blocks.19.layernorm1',
                                             'backbone.encoder.blocks.19.layernorm2',
                                             'backbone.encoder.blocks.20.layernorm1',
                                             'backbone.encoder.blocks.20.layernorm2',
                                             'backbone.encoder.blocks.21.layernorm1',
                                             'backbone.encoder.blocks.21.layernorm2',
                                             'backbone.encoder.blocks.22.layernorm1',
                                             'backbone.encoder.blocks.22.layernorm2',
                                             'backbone.encoder.blocks.23.layernorm1',
                                             'backbone.encoder.blocks.23.layernorm2',
                                             'backbone.layernorm']],

            'backbone.get_attention_mask': ['INPUT', ['backbone.encoder.blocks.0.attention.softmax',
                                                      'backbone.encoder.blocks.1.attention.softmax',
                                                      'backbone.encoder.blocks.2.attention.softmax',
                                                      'backbone.encoder.blocks.3.attention.softmax',
                                                      'backbone.encoder.blocks.4.attention.softmax',
                                                      'backbone.encoder.blocks.5.attention.softmax',
                                                      'backbone.encoder.blocks.6.attention.softmax',
                                                      'backbone.encoder.blocks.7.attention.softmax',
                                                      'backbone.encoder.blocks.8.attention.softmax',
                                                      'backbone.encoder.blocks.9.attention.softmax',
                                                      'backbone.encoder.blocks.10.attention.softmax',
                                                      'backbone.encoder.blocks.11.attention.softmax',
                                                      'backbone.encoder.blocks.12.attention.softmax',
                                                      'backbone.encoder.blocks.13.attention.softmax',
                                                      'backbone.encoder.blocks.14.attention.softmax',
                                                      'backbone.encoder.blocks.15.attention.softmax',
                                                      'backbone.encoder.blocks.16.attention.softmax',
                                                      'backbone.encoder.blocks.17.attention.softmax',
                                                      'backbone.encoder.blocks.18.attention.softmax',
                                                      'backbone.encoder.blocks.19.attention.softmax',
                                                      'backbone.encoder.blocks.20.attention.softmax',
                                                      'backbone.encoder.blocks.21.attention.softmax',
                                                      'backbone.encoder.blocks.22.attention.softmax',
                                                      'backbone.encoder.blocks.23.attention.softmax',
                                                      ]],

            # ###
            'backbone.encoder.blocks.0.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding'],
                                                     ['backbone.encoder.blocks.0.attention.dense1',
                                                      'backbone.encoder.blocks.0.attention.dense2',
                                                      'backbone.encoder.blocks.0.attention.dense3']],

            'backbone.encoder.blocks.0.attention.dense1': ['backbone.encoder.blocks.0.layernorm1',
                                                           'backbone.encoder.blocks.0.attention.softmax'],
            'backbone.encoder.blocks.0.attention.dense2': ['backbone.encoder.blocks.0.layernorm1',
                                                           'backbone.encoder.blocks.0.attention.softmax'],
            'backbone.encoder.blocks.0.attention.dense3': ['backbone.encoder.blocks.0.layernorm1',
                                                           'backbone.encoder.blocks.0.attention.projection'],

            'backbone.encoder.blocks.0.attention.softmax': [
                ['backbone.encoder.blocks.0.attention.dense1', 'backbone.encoder.blocks.0.attention.dense2',
                 'backbone.get_attention_mask'],
                'backbone.encoder.blocks.0.attention.prob_dropout'],

            'backbone.encoder.blocks.0.attention.prob_dropout': ['backbone.encoder.blocks.0.attention.softmax',
                                                                 'backbone.encoder.blocks.0.attention.projection'],

            'backbone.encoder.blocks.0.attention.projection': [
                ['backbone.encoder.blocks.0.attention.dense3', 'backbone.encoder.blocks.0.attention.prob_dropout'],
                'backbone.encoder.blocks.0.attention.dropout'],
            'backbone.encoder.blocks.0.attention.dropout': ['backbone.encoder.blocks.0.attention.projection',
                                                            ['backbone.encoder.blocks.0.layernorm2',
                                                             'backbone.encoder.blocks.1.layernorm1',
                                                             'backbone.encoder.blocks.1.layernorm2',
                                                             'backbone.encoder.blocks.2.layernorm1',
                                                             'backbone.encoder.blocks.2.layernorm2',
                                                             'backbone.encoder.blocks.3.layernorm1',
                                                             'backbone.encoder.blocks.3.layernorm2',
                                                             'backbone.encoder.blocks.4.layernorm1',
                                                             'backbone.encoder.blocks.4.layernorm2',
                                                             'backbone.encoder.blocks.5.layernorm1',
                                                             'backbone.encoder.blocks.5.layernorm2',
                                                             'backbone.encoder.blocks.6.layernorm1',
                                                             'backbone.encoder.blocks.6.layernorm2',
                                                             'backbone.encoder.blocks.7.layernorm1',
                                                             'backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm'
                                                             ]],

            'backbone.encoder.blocks.0.layernorm2': [
                ['backbone.encoder.blocks.0.attention.dropout', 'backbone.word_embedding',
                 'backbone.position_embedding'],
                'backbone.encoder.blocks.0.output.hidden_act'],
            'backbone.encoder.blocks.0.output.hidden_act': ['backbone.encoder.blocks.0.layernorm2',
                                                            'backbone.encoder.blocks.0.output.mapping'],
            'backbone.encoder.blocks.0.output.mapping': ['backbone.encoder.blocks.0.output.hidden_act',
                                                         'backbone.encoder.blocks.0.output.projection'],
            'backbone.encoder.blocks.0.output.projection': ['backbone.encoder.blocks.0.output.mapping',
                                                            'backbone.encoder.blocks.0.output.dropout'],
            'backbone.encoder.blocks.0.output.dropout': ['backbone.encoder.blocks.0.output.projection',
                                                         ['backbone.encoder.blocks.1.layernorm1',
                                                          'backbone.encoder.blocks.1.layernorm2',
                                                          'backbone.encoder.blocks.2.layernorm1',
                                                          'backbone.encoder.blocks.2.layernorm2',
                                                          'backbone.encoder.blocks.3.layernorm1',
                                                          'backbone.encoder.blocks.3.layernorm2',
                                                          'backbone.encoder.blocks.4.layernorm1',
                                                          'backbone.encoder.blocks.4.layernorm2',
                                                          'backbone.encoder.blocks.5.layernorm1',
                                                          'backbone.encoder.blocks.5.layernorm2',
                                                          'backbone.encoder.blocks.6.layernorm1',
                                                          'backbone.encoder.blocks.6.layernorm2',
                                                          'backbone.encoder.blocks.7.layernorm1',
                                                          'backbone.encoder.blocks.7.layernorm2',
                                                          'backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm'
                                                          ]],

            # ####
            'backbone.encoder.blocks.1.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout'],
                                                     ['backbone.encoder.blocks.1.attention.dense1',
                                                      'backbone.encoder.blocks.1.attention.dense2',
                                                      'backbone.encoder.blocks.1.attention.dense3']],

            'backbone.encoder.blocks.1.attention.dense1': ['backbone.encoder.blocks.1.layernorm1',
                                                           'backbone.encoder.blocks.1.attention.softmax'],
            'backbone.encoder.blocks.1.attention.dense2': ['backbone.encoder.blocks.1.layernorm1',
                                                           'backbone.encoder.blocks.1.attention.softmax'],
            'backbone.encoder.blocks.1.attention.dense3': ['backbone.encoder.blocks.1.layernorm1',
                                                           'backbone.encoder.blocks.1.attention.projection'],

            'backbone.encoder.blocks.1.attention.softmax': [
                ['backbone.encoder.blocks.1.attention.dense1', 'backbone.encoder.blocks.1.attention.dense2',
                 'backbone.get_attention_mask'],
                'backbone.encoder.blocks.1.attention.prob_dropout'],

            'backbone.encoder.blocks.1.attention.prob_dropout': ['backbone.encoder.blocks.1.attention.softmax',
                                                                 'backbone.encoder.blocks.1.attention.projection'],

            'backbone.encoder.blocks.1.attention.projection': [
                ['backbone.encoder.blocks.1.attention.dense3', 'backbone.encoder.blocks.1.attention.prob_dropout'],
                'backbone.encoder.blocks.1.attention.dropout'],
            'backbone.encoder.blocks.1.attention.dropout': ['backbone.encoder.blocks.1.attention.projection',
                                                            ['backbone.encoder.blocks.1.layernorm2',
                                                             'backbone.encoder.blocks.2.layernorm1',
                                                             'backbone.encoder.blocks.2.layernorm2',
                                                             'backbone.encoder.blocks.3.layernorm1',
                                                             'backbone.encoder.blocks.3.layernorm2',
                                                             'backbone.encoder.blocks.4.layernorm1',
                                                             'backbone.encoder.blocks.4.layernorm2',
                                                             'backbone.encoder.blocks.5.layernorm1',
                                                             'backbone.encoder.blocks.5.layernorm2',
                                                             'backbone.encoder.blocks.6.layernorm1',
                                                             'backbone.encoder.blocks.6.layernorm2',
                                                             'backbone.encoder.blocks.7.layernorm1',
                                                             'backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm', ]],

            'backbone.encoder.blocks.1.layernorm2': [
                ['backbone.word_embedding', 'backbone.position_embedding',
                 'backbone.encoder.blocks.0.attention.dropout', 'backbone.encoder.blocks.0.output.dropout',
                 'backbone.encoder.blocks.1.attention.dropout', ],
                'backbone.encoder.blocks.1.output.mapping'],
            'backbone.encoder.blocks.1.output.mapping': ['backbone.encoder.blocks.1.layernorm2',
                                                         'backbone.encoder.blocks.1.output.projection'],
            'backbone.encoder.blocks.1.output.projection': ['backbone.encoder.blocks.1.output.mapping',
                                                            'backbone.encoder.blocks.1.output.dropout'],
            'backbone.encoder.blocks.1.output.dropout': ['backbone.encoder.blocks.1.output.projection',
                                                         ['backbone.encoder.blocks.2.layernorm1',
                                                          'backbone.encoder.blocks.2.layernorm2',
                                                          'backbone.encoder.blocks.3.layernorm1',
                                                          'backbone.encoder.blocks.3.layernorm2',
                                                          'backbone.encoder.blocks.4.layernorm1',
                                                          'backbone.encoder.blocks.4.layernorm2',
                                                          'backbone.encoder.blocks.5.layernorm1',
                                                          'backbone.encoder.blocks.5.layernorm2',
                                                          'backbone.encoder.blocks.6.layernorm1',
                                                          'backbone.encoder.blocks.6.layernorm2',
                                                          'backbone.encoder.blocks.7.layernorm1',
                                                          'backbone.encoder.blocks.7.layernorm2',
                                                          'backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.2.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.2.attention.dense1',
                                                      'backbone.encoder.blocks.2.attention.dense2',
                                                      'backbone.encoder.blocks.2.attention.dense3']],

            'backbone.encoder.blocks.2.attention.dense1': ['backbone.encoder.blocks.2.layernorm1',
                                                           'backbone.encoder.blocks.2.attention.softmax'],
            'backbone.encoder.blocks.2.attention.dense2': ['backbone.encoder.blocks.2.layernorm1',
                                                           'backbone.encoder.blocks.2.attention.softmax'],
            'backbone.encoder.blocks.2.attention.dense3': ['backbone.encoder.blocks.2.layernorm1',
                                                           'backbone.encoder.blocks.2.attention.projection'],

            'backbone.encoder.blocks.2.attention.softmax': [
                ['backbone.encoder.blocks.2.attention.dense1', 'backbone.encoder.blocks.2.attention.dense2',
                 'backbone.get_attention_mask'],
                'backbone.encoder.blocks.2.attention.prob_dropout'],

            'backbone.encoder.blocks.2.attention.prob_dropout': ['backbone.encoder.blocks.2.attention.softmax',
                                                                 'backbone.encoder.blocks.2.attention.projection'],

            'backbone.encoder.blocks.2.attention.projection': [
                ['backbone.encoder.blocks.2.attention.dense3', 'backbone.encoder.blocks.2.attention.prob_dropout'],
                'backbone.encoder.blocks.2.attention.dropout'],
            'backbone.encoder.blocks.2.attention.dropout': ['backbone.encoder.blocks.2.attention.projection',
                                                            ['backbone.encoder.blocks.2.layernorm2',
                                                             'backbone.encoder.blocks.3.layernorm1',
                                                             'backbone.encoder.blocks.3.layernorm2',
                                                             'backbone.encoder.blocks.4.layernorm1',
                                                             'backbone.encoder.blocks.4.layernorm2',
                                                             'backbone.encoder.blocks.5.layernorm1',
                                                             'backbone.encoder.blocks.5.layernorm2',
                                                             'backbone.encoder.blocks.6.layernorm1',
                                                             'backbone.encoder.blocks.6.layernorm2',
                                                             'backbone.encoder.blocks.7.layernorm1',
                                                             'backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm']],

            'backbone.encoder.blocks.2.layernorm2': [
                ['backbone.word_embedding', 'backbone.position_embedding',
                 'backbone.encoder.blocks.0.attention.dropout', 'backbone.encoder.blocks.0.output.dropout',
                 'backbone.encoder.blocks.1.attention.dropout',
                 'backbone.encoder.blocks.1.output.dropout',
                 'backbone.encoder.blocks.2.attention.dropout', ],
                'backbone.encoder.blocks.2.output.mapping'],
            'backbone.encoder.blocks.2.output.mapping': ['backbone.encoder.blocks.2.layernorm2',
                                                         'backbone.encoder.blocks.2.output.projection'],
            'backbone.encoder.blocks.2.output.projection': ['backbone.encoder.blocks.2.output.mapping',
                                                            'backbone.encoder.blocks.2.output.dropout'],
            'backbone.encoder.blocks.2.output.dropout': ['backbone.encoder.blocks.2.output.projection',
                                                         ['backbone.encoder.blocks.3.layernorm1',
                                                          'backbone.encoder.blocks.3.layernorm2',
                                                          'backbone.encoder.blocks.4.layernorm1',
                                                          'backbone.encoder.blocks.4.layernorm2',
                                                          'backbone.encoder.blocks.5.layernorm1',
                                                          'backbone.encoder.blocks.5.layernorm2',
                                                          'backbone.encoder.blocks.6.layernorm1',
                                                          'backbone.encoder.blocks.6.layernorm2',
                                                          'backbone.encoder.blocks.7.layernorm1',
                                                          'backbone.encoder.blocks.7.layernorm2',
                                                          'backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.3.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.3.attention.dense1',
                                                      'backbone.encoder.blocks.3.attention.dense2',
                                                      'backbone.encoder.blocks.3.attention.dense3']],

            'backbone.encoder.blocks.3.attention.dense1': ['backbone.encoder.blocks.3.layernorm1',
                                                           'backbone.encoder.blocks.3.attention.softmax'],
            'backbone.encoder.blocks.3.attention.dense2': ['backbone.encoder.blocks.3.layernorm1',
                                                           'backbone.encoder.blocks.3.attention.softmax'],
            'backbone.encoder.blocks.3.attention.dense3': ['backbone.encoder.blocks.3.layernorm1',
                                                           'backbone.encoder.blocks.3.attention.projection'],

            'backbone.encoder.blocks.3.attention.softmax': [['backbone.encoder.blocks.3.attention.dense1',
                                                             'backbone.encoder.blocks.3.attention.dense2',
                                                             'backbone.get_attention_mask'],
                                                            'backbone.encoder.blocks.3.attention.prob_dropout'],

            'backbone.encoder.blocks.3.attention.prob_dropout': ['backbone.encoder.blocks.3.attention.softmax',
                                                                 'backbone.encoder.blocks.3.attention.projection'],

            'backbone.encoder.blocks.3.attention.projection': [['backbone.encoder.blocks.3.attention.dense3',
                                                                'backbone.encoder.blocks.3.attention.prob_dropout'],
                                                               'backbone.encoder.blocks.3.attention.dropout'],
            'backbone.encoder.blocks.3.attention.dropout': ['backbone.encoder.blocks.3.attention.projection',
                                                            ['backbone.encoder.blocks.3.layernorm2',
                                                             'backbone.encoder.blocks.4.layernorm1',
                                                             'backbone.encoder.blocks.4.layernorm2',
                                                             'backbone.encoder.blocks.5.layernorm1',
                                                             'backbone.encoder.blocks.5.layernorm2',
                                                             'backbone.encoder.blocks.6.layernorm1',
                                                             'backbone.encoder.blocks.6.layernorm2',
                                                             'backbone.encoder.blocks.7.layernorm1',
                                                             'backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm']],

            'backbone.encoder.blocks.3.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      ],
                                                     'backbone.encoder.blocks.3.output.mapping'],
            'backbone.encoder.blocks.3.output.mapping': ['backbone.encoder.blocks.3.layernorm2',
                                                         'backbone.encoder.blocks.3.output.projection'],
            'backbone.encoder.blocks.3.output.projection': ['backbone.encoder.blocks.3.output.mapping',
                                                            'backbone.encoder.blocks.3.output.dropout'],
            'backbone.encoder.blocks.3.output.dropout': ['backbone.encoder.blocks.3.output.projection',
                                                         ['backbone.encoder.blocks.4.layernorm1',
                                                          'backbone.encoder.blocks.4.layernorm2',
                                                          'backbone.encoder.blocks.5.layernorm1',
                                                          'backbone.encoder.blocks.5.layernorm2',
                                                          'backbone.encoder.blocks.6.layernorm1',
                                                          'backbone.encoder.blocks.6.layernorm2',
                                                          'backbone.encoder.blocks.7.layernorm1',
                                                          'backbone.encoder.blocks.7.layernorm2',
                                                          'backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.4.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.4.attention.dense1',
                                                      'backbone.encoder.blocks.4.attention.dense2',
                                                      'backbone.encoder.blocks.4.attention.dense3']],
            'backbone.encoder.blocks.4.attention.dense1': ['backbone.encoder.blocks.4.layernorm1',
                                                           'backbone.encoder.blocks.4.attention.softmax'],
            'backbone.encoder.blocks.4.attention.dense2': ['backbone.encoder.blocks.4.layernorm1',
                                                           'backbone.encoder.blocks.4.attention.softmax'],
            'backbone.encoder.blocks.4.attention.dense3': ['backbone.encoder.blocks.4.layernorm1',
                                                           'backbone.encoder.blocks.4.attention.projection'],

            'backbone.encoder.blocks.4.attention.softmax': [['backbone.encoder.blocks.4.attention.dense1',
                                                             'backbone.encoder.blocks.4.attention.dense2',
                                                             'backbone.get_attention_mask'],
                                                            'backbone.encoder.blocks.4.attention.prob_dropout'],

            'backbone.encoder.blocks.4.attention.prob_dropout': ['backbone.encoder.blocks.4.attention.softmax',
                                                                 'backbone.encoder.blocks.4.attention.projection'],

            'backbone.encoder.blocks.4.attention.projection': [['backbone.encoder.blocks.4.attention.dense3',
                                                                'backbone.encoder.blocks.4.attention.prob_dropout'],
                                                               'backbone.encoder.blocks.4.attention.dropout'],
            'backbone.encoder.blocks.4.attention.dropout': ['backbone.encoder.blocks.4.attention.projection',
                                                            ['backbone.encoder.blocks.4.layernorm2',
                                                             'backbone.encoder.blocks.5.layernorm1',
                                                             'backbone.encoder.blocks.5.layernorm2',
                                                             'backbone.encoder.blocks.6.layernorm1',
                                                             'backbone.encoder.blocks.6.layernorm2',
                                                             'backbone.encoder.blocks.7.layernorm1',
                                                             'backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm'
                                                             ]],

            'backbone.encoder.blocks.4.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      ],
                                                     'backbone.encoder.blocks.4.output.mapping'],
            'backbone.encoder.blocks.4.output.mapping': ['backbone.encoder.blocks.4.layernorm2',
                                                         'backbone.encoder.blocks.4.output.projection'],
            'backbone.encoder.blocks.4.output.projection': ['backbone.encoder.blocks.4.output.mapping',
                                                            'backbone.encoder.blocks.4.output.dropout'],
            'backbone.encoder.blocks.4.output.dropout': ['backbone.encoder.blocks.4.output.projection',
                                                         ['backbone.encoder.blocks.5.layernorm1',
                                                          'backbone.encoder.blocks.5.layernorm2',
                                                          'backbone.encoder.blocks.6.layernorm1',
                                                          'backbone.encoder.blocks.6.layernorm2',
                                                          'backbone.encoder.blocks.7.layernorm1',
                                                          'backbone.encoder.blocks.7.layernorm2',
                                                          'backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.5.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.5.attention.dense1',
                                                      'backbone.encoder.blocks.5.attention.dense2',
                                                      'backbone.encoder.blocks.5.attention.dense3']],
            'backbone.encoder.blocks.5.attention.dense1': ['backbone.encoder.blocks.5.layernorm1',
                                                           'backbone.encoder.blocks.5.attention.softmax'],
            'backbone.encoder.blocks.5.attention.dense2': ['backbone.encoder.blocks.5.layernorm1',
                                                           'backbone.encoder.blocks.5.attention.softmax'],
            'backbone.encoder.blocks.5.attention.dense3': ['backbone.encoder.blocks.5.layernorm1',
                                                           'backbone.encoder.blocks.5.attention.projection'],

            'backbone.encoder.blocks.5.attention.softmax': [['backbone.encoder.blocks.5.attention.dense1',
                                                             'backbone.encoder.blocks.5.attention.dense2',
                                                             'backbone.get_attention_mask'],
                                                            'backbone.encoder.blocks.5.attention.prob_dropout'],

            'backbone.encoder.blocks.5.attention.prob_dropout': ['backbone.encoder.blocks.5.attention.softmax',
                                                                 'backbone.encoder.blocks.5.attention.projection'],

            'backbone.encoder.blocks.5.attention.projection': [['backbone.encoder.blocks.5.attention.dense3',
                                                                'backbone.encoder.blocks.5.attention.prob_dropout'],
                                                               'backbone.encoder.blocks.5.attention.dropout'],
            'backbone.encoder.blocks.5.attention.dropout': ['backbone.encoder.blocks.5.attention.projection',
                                                            ['backbone.encoder.blocks.5.layernorm2',
                                                             'backbone.encoder.blocks.6.layernorm1',
                                                             'backbone.encoder.blocks.6.layernorm2',
                                                             'backbone.encoder.blocks.7.layernorm1',
                                                             'backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm']],

            'backbone.encoder.blocks.5.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      ],
                                                     'backbone.encoder.blocks.5.output.mapping'],
            'backbone.encoder.blocks.5.output.mapping': ['backbone.encoder.blocks.5.layernorm2',
                                                         'backbone.encoder.blocks.5.output.projection'],
            'backbone.encoder.blocks.5.output.projection': ['backbone.encoder.blocks.5.output.mapping',
                                                            'backbone.encoder.blocks.5.output.dropout'],
            'backbone.encoder.blocks.5.output.dropout': ['backbone.encoder.blocks.5.output.projection',
                                                         ['backbone.encoder.blocks.6.layernorm1',
                                                          'backbone.encoder.blocks.6.layernorm2',
                                                          'backbone.encoder.blocks.7.layernorm1',
                                                          'backbone.encoder.blocks.7.layernorm2',
                                                          'backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.6.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.6.attention.dense1',
                                                      'backbone.encoder.blocks.6.attention.dense2',
                                                      'backbone.encoder.blocks.6.attention.dense3']],
            'backbone.encoder.blocks.6.attention.dense1': ['backbone.encoder.blocks.6.layernorm1',
                                                           'backbone.encoder.blocks.6.attention.softmax'],
            'backbone.encoder.blocks.6.attention.dense2': ['backbone.encoder.blocks.6.layernorm1',
                                                           'backbone.encoder.blocks.6.attention.softmax'],
            'backbone.encoder.blocks.6.attention.dense3': ['backbone.encoder.blocks.6.layernorm1',
                                                           'backbone.encoder.blocks.6.attention.projection'],

            'backbone.encoder.blocks.6.attention.softmax': [['backbone.encoder.blocks.6.attention.dense1',
                                                             'backbone.encoder.blocks.6.attention.dense2',
                                                             'backbone.get_attention_mask'],
                                                            'backbone.encoder.blocks.6.attention.prob_dropout'],

            'backbone.encoder.blocks.6.attention.prob_dropout': ['backbone.encoder.blocks.6.attention.softmax',
                                                                 'backbone.encoder.blocks.6.attention.projection'],

            'backbone.encoder.blocks.6.attention.projection': [['backbone.encoder.blocks.6.attention.dense3',
                                                                'backbone.encoder.blocks.6.attention.prob_dropout'],
                                                               'backbone.encoder.blocks.6.attention.dropout'],
            'backbone.encoder.blocks.6.attention.dropout': ['backbone.encoder.blocks.6.attention.projection',
                                                            ['backbone.encoder.blocks.6.layernorm2',
                                                             'backbone.encoder.blocks.7.layernorm1',
                                                             'backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm']],

            'backbone.encoder.blocks.6.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      'backbone.encoder.blocks.6.attention.dropout',
                                                      ],
                                                     'backbone.encoder.blocks.6.output.mapping'],
            'backbone.encoder.blocks.6.output.mapping': ['backbone.encoder.blocks.6.layernorm2',
                                                         'backbone.encoder.blocks.6.output.projection'],
            'backbone.encoder.blocks.6.output.projection': ['backbone.encoder.blocks.6.output.mapping',
                                                            'backbone.encoder.blocks.6.output.dropout'],
            'backbone.encoder.blocks.6.output.dropout': ['backbone.encoder.blocks.6.output.projection',
                                                         ['backbone.encoder.blocks.7.layernorm1',
                                                          'backbone.encoder.blocks.7.layernorm2',
                                                          'backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.7.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      'backbone.encoder.blocks.6.attention.dropout',
                                                      'backbone.encoder.blocks.6.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.7.attention.dense1',
                                                      'backbone.encoder.blocks.7.attention.dense2',
                                                      'backbone.encoder.blocks.7.attention.dense3']],
            'backbone.encoder.blocks.7.attention.dense1': ['backbone.encoder.blocks.7.layernorm1',
                                                           'backbone.encoder.blocks.7.attention.softmax'],
            'backbone.encoder.blocks.7.attention.dense2': ['backbone.encoder.blocks.7.layernorm1',
                                                           'backbone.encoder.blocks.7.attention.softmax'],
            'backbone.encoder.blocks.7.attention.dense3': ['backbone.encoder.blocks.7.layernorm1',
                                                           'backbone.encoder.blocks.7.attention.projection'],

            'backbone.encoder.blocks.7.attention.softmax': [['backbone.encoder.blocks.7.attention.dense1',
                                                             'backbone.encoder.blocks.7.attention.dense2',
                                                             'backbone.get_attention_mask'],
                                                            'backbone.encoder.blocks.7.attention.prob_dropout'],

            'backbone.encoder.blocks.7.attention.prob_dropout': ['backbone.encoder.blocks.7.attention.softmax',
                                                                 'backbone.encoder.blocks.7.attention.projection'],

            'backbone.encoder.blocks.7.attention.projection': [['backbone.encoder.blocks.7.attention.dense3',
                                                                'backbone.encoder.blocks.7.attention.prob_dropout'],
                                                               'backbone.encoder.blocks.7.attention.dropout'],
            'backbone.encoder.blocks.7.attention.dropout': ['backbone.encoder.blocks.7.attention.projection',
                                                            ['backbone.encoder.blocks.7.layernorm2',
                                                             'backbone.encoder.blocks.8.layernorm1',
                                                             'backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm']],

            'backbone.encoder.blocks.7.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      'backbone.encoder.blocks.6.attention.dropout',
                                                      'backbone.encoder.blocks.6.output.dropout',
                                                      'backbone.encoder.blocks.7.attention.dropout',
                                                      ],
                                                     'backbone.encoder.blocks.7.output.mapping'],
            'backbone.encoder.blocks.7.output.mapping': ['backbone.encoder.blocks.7.layernorm2',
                                                         'backbone.encoder.blocks.7.output.projection'],
            'backbone.encoder.blocks.7.output.projection': ['backbone.encoder.blocks.7.output.mapping',
                                                            'backbone.encoder.blocks.7.output.dropout'],
            'backbone.encoder.blocks.7.output.dropout': ['backbone.encoder.blocks.7.output.projection',
                                                         ['backbone.encoder.blocks.8.layernorm1',
                                                          'backbone.encoder.blocks.8.layernorm2',
                                                          'backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.8.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      'backbone.encoder.blocks.6.attention.dropout',
                                                      'backbone.encoder.blocks.6.output.dropout',
                                                      'backbone.encoder.blocks.7.attention.dropout',
                                                      'backbone.encoder.blocks.7.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.8.attention.dense1',
                                                      'backbone.encoder.blocks.8.attention.dense2',
                                                      'backbone.encoder.blocks.8.attention.dense3']],
            'backbone.encoder.blocks.8.attention.dense1': ['backbone.encoder.blocks.8.layernorm1',
                                                           'backbone.encoder.blocks.8.attention.softmax'],
            'backbone.encoder.blocks.8.attention.dense2': ['backbone.encoder.blocks.8.layernorm1',
                                                           'backbone.encoder.blocks.8.attention.softmax'],
            'backbone.encoder.blocks.8.attention.dense3': ['backbone.encoder.blocks.8.layernorm1',
                                                           'backbone.encoder.blocks.8.attention.projection'],

            'backbone.encoder.blocks.8.attention.softmax': [['backbone.encoder.blocks.8.attention.dense1',
                                                             'backbone.encoder.blocks.8.attention.dense2',
                                                             'backbone.get_attention_mask'],
                                                            'backbone.encoder.blocks.8.attention.prob_dropout'],

            'backbone.encoder.blocks.8.attention.prob_dropout': ['backbone.encoder.blocks.8.attention.softmax',
                                                                 'backbone.encoder.blocks.8.attention.projection'],

            'backbone.encoder.blocks.8.attention.projection': [['backbone.encoder.blocks.8.attention.dense3',
                                                                'backbone.encoder.blocks.8.attention.prob_dropout'],
                                                               'backbone.encoder.blocks.8.attention.dropout'],
            'backbone.encoder.blocks.8.attention.dropout': ['backbone.encoder.blocks.8.attention.projection',
                                                            ['backbone.encoder.blocks.8.layernorm2',
                                                             'backbone.encoder.blocks.9.layernorm1',
                                                             'backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm']],

            'backbone.encoder.blocks.8.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      'backbone.encoder.blocks.6.attention.dropout',
                                                      'backbone.encoder.blocks.6.output.dropout',
                                                      'backbone.encoder.blocks.7.attention.dropout',
                                                      'backbone.encoder.blocks.7.output.dropout',
                                                      'backbone.encoder.blocks.8.attention.dropout',
                                                      ],
                                                     'backbone.encoder.blocks.8.output.mapping'],
            'backbone.encoder.blocks.8.output.mapping': ['backbone.encoder.blocks.8.layernorm2',
                                                         'backbone.encoder.blocks.8.output.projection'],
            'backbone.encoder.blocks.8.output.projection': ['backbone.encoder.blocks.8.output.mapping',
                                                            'backbone.encoder.blocks.8.output.dropout'],
            'backbone.encoder.blocks.8.output.dropout': ['backbone.encoder.blocks.8.output.projection',
                                                         ['backbone.encoder.blocks.9.layernorm1',
                                                          'backbone.encoder.blocks.9.layernorm2',
                                                          'backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.9.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      'backbone.encoder.blocks.6.attention.dropout',
                                                      'backbone.encoder.blocks.6.output.dropout',
                                                      'backbone.encoder.blocks.7.attention.dropout',
                                                      'backbone.encoder.blocks.7.output.dropout',
                                                      'backbone.encoder.blocks.8.attention.dropout',
                                                      'backbone.encoder.blocks.8.output.dropout',
                                                      ],
                                                     ['backbone.encoder.blocks.9.attention.dense1',
                                                      'backbone.encoder.blocks.9.attention.dense2',
                                                      'backbone.encoder.blocks.9.attention.dense3']],
            'backbone.encoder.blocks.9.attention.dense1': ['backbone.encoder.blocks.9.layernorm1',
                                                           'backbone.encoder.blocks.9.attention.softmax'],
            'backbone.encoder.blocks.9.attention.dense2': ['backbone.encoder.blocks.9.layernorm1',
                                                           'backbone.encoder.blocks.9.attention.softmax'],
            'backbone.encoder.blocks.9.attention.dense3': ['backbone.encoder.blocks.9.layernorm1',
                                                           'backbone.encoder.blocks.9.attention.projection'],

            'backbone.encoder.blocks.9.attention.softmax': [['backbone.encoder.blocks.9.attention.dense1',
                                                             'backbone.encoder.blocks.9.attention.dense2',
                                                             'backbone.get_attention_mask'],
                                                            'backbone.encoder.blocks.9.attention.prob_dropout'],

            'backbone.encoder.blocks.9.attention.prob_dropout': ['backbone.encoder.blocks.9.attention.softmax',
                                                                 'backbone.encoder.blocks.9.attention.projection'],

            'backbone.encoder.blocks.9.attention.projection': [['backbone.encoder.blocks.9.attention.dense3',
                                                                'backbone.encoder.blocks.9.attention.prob_dropout'],
                                                               'backbone.encoder.blocks.9.attention.dropout'],
            'backbone.encoder.blocks.9.attention.dropout': ['backbone.encoder.blocks.9.attention.projection',
                                                            ['backbone.encoder.blocks.9.layernorm2',
                                                             'backbone.encoder.blocks.10.layernorm1',
                                                             'backbone.encoder.blocks.10.layernorm2',
                                                             'backbone.encoder.blocks.11.layernorm1',
                                                             'backbone.encoder.blocks.11.layernorm2',
                                                             'backbone.encoder.blocks.12.layernorm1',
                                                             'backbone.encoder.blocks.12.layernorm2',
                                                             'backbone.encoder.blocks.13.layernorm1',
                                                             'backbone.encoder.blocks.13.layernorm2',
                                                             'backbone.encoder.blocks.14.layernorm1',
                                                             'backbone.encoder.blocks.14.layernorm2',
                                                             'backbone.encoder.blocks.15.layernorm1',
                                                             'backbone.encoder.blocks.15.layernorm2',
                                                             'backbone.encoder.blocks.16.layernorm1',
                                                             'backbone.encoder.blocks.16.layernorm2',
                                                             'backbone.encoder.blocks.17.layernorm1',
                                                             'backbone.encoder.blocks.17.layernorm2',
                                                             'backbone.encoder.blocks.18.layernorm1',
                                                             'backbone.encoder.blocks.18.layernorm2',
                                                             'backbone.encoder.blocks.19.layernorm1',
                                                             'backbone.encoder.blocks.19.layernorm2',
                                                             'backbone.encoder.blocks.20.layernorm1',
                                                             'backbone.encoder.blocks.20.layernorm2',
                                                             'backbone.encoder.blocks.21.layernorm1',
                                                             'backbone.encoder.blocks.21.layernorm2',
                                                             'backbone.encoder.blocks.22.layernorm1',
                                                             'backbone.encoder.blocks.22.layernorm2',
                                                             'backbone.encoder.blocks.23.layernorm1',
                                                             'backbone.encoder.blocks.23.layernorm2',
                                                             'backbone.layernorm']],

            'backbone.encoder.blocks.9.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                      'backbone.encoder.blocks.0.attention.dropout',
                                                      'backbone.encoder.blocks.0.output.dropout',
                                                      'backbone.encoder.blocks.1.attention.dropout',
                                                      'backbone.encoder.blocks.1.output.dropout',
                                                      'backbone.encoder.blocks.2.attention.dropout',
                                                      'backbone.encoder.blocks.2.output.dropout',
                                                      'backbone.encoder.blocks.3.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.3.output.dropout',
                                                      'backbone.encoder.blocks.4.attention.dropout',
                                                      'backbone.encoder.blocks.4.output.dropout',
                                                      'backbone.encoder.blocks.5.attention.dropout',
                                                      'backbone.encoder.blocks.5.output.dropout',
                                                      'backbone.encoder.blocks.6.attention.dropout',
                                                      'backbone.encoder.blocks.6.output.dropout',
                                                      'backbone.encoder.blocks.7.attention.dropout',
                                                      'backbone.encoder.blocks.7.output.dropout',
                                                      'backbone.encoder.blocks.8.attention.dropout',
                                                      'backbone.encoder.blocks.8.output.dropout',
                                                      'backbone.encoder.blocks.9.attention.dropout',
                                                      ],
                                                     'backbone.encoder.blocks.9.output.mapping'],
            'backbone.encoder.blocks.9.output.mapping': ['backbone.encoder.blocks.9.layernorm2',
                                                         'backbone.encoder.blocks.9.output.projection'],
            'backbone.encoder.blocks.9.output.projection': ['backbone.encoder.blocks.9.output.mapping',
                                                            'backbone.encoder.blocks.9.output.dropout'],
            'backbone.encoder.blocks.9.output.dropout': ['backbone.encoder.blocks.9.output.projection',
                                                         ['backbone.encoder.blocks.10.layernorm1',
                                                          'backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.11.layernorm1',
                                                          'backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.12.layernorm1',
                                                          'backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.13.layernorm1',
                                                          'backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.14.layernorm1',
                                                          'backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.15.layernorm1',
                                                          'backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.16.layernorm1',
                                                          'backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.17.layernorm1',
                                                          'backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.18.layernorm1',
                                                          'backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.19.layernorm1',
                                                          'backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.20.layernorm1',
                                                          'backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.21.layernorm1',
                                                          'backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.22.layernorm1',
                                                          'backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.23.layernorm1',
                                                          'backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.10.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.10.attention.dense1',
                                                       'backbone.encoder.blocks.10.attention.dense2',
                                                       'backbone.encoder.blocks.10.attention.dense3']],
            'backbone.encoder.blocks.10.attention.dense1': ['backbone.encoder.blocks.10.layernorm1',
                                                            'backbone.encoder.blocks.10.attention.softmax'],
            'backbone.encoder.blocks.10.attention.dense2': ['backbone.encoder.blocks.10.layernorm1',
                                                            'backbone.encoder.blocks.10.attention.softmax'],
            'backbone.encoder.blocks.10.attention.dense3': ['backbone.encoder.blocks.10.layernorm1',
                                                            'backbone.encoder.blocks.10.attention.projection'],

            'backbone.encoder.blocks.10.attention.softmax': [['backbone.encoder.blocks.10.attention.dense1',
                                                              'backbone.encoder.blocks.10.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.10.attention.prob_dropout'],

            'backbone.encoder.blocks.10.attention.prob_dropout': ['backbone.encoder.blocks.10.attention.softmax',
                                                                  'backbone.encoder.blocks.10.attention.projection'],

            'backbone.encoder.blocks.10.attention.projection': [['backbone.encoder.blocks.10.attention.dense3',
                                                                 'backbone.encoder.blocks.10.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.10.attention.dropout'],
            'backbone.encoder.blocks.10.attention.dropout': ['backbone.encoder.blocks.10.attention.projection',
                                                             ['backbone.encoder.blocks.10.layernorm2',
                                                              'backbone.encoder.blocks.11.layernorm1',
                                                              'backbone.encoder.blocks.11.layernorm2',
                                                              'backbone.encoder.blocks.12.layernorm1',
                                                              'backbone.encoder.blocks.12.layernorm2',
                                                              'backbone.encoder.blocks.13.layernorm1',
                                                              'backbone.encoder.blocks.13.layernorm2',
                                                              'backbone.encoder.blocks.14.layernorm1',
                                                              'backbone.encoder.blocks.14.layernorm2',
                                                              'backbone.encoder.blocks.15.layernorm1',
                                                              'backbone.encoder.blocks.15.layernorm2',
                                                              'backbone.encoder.blocks.16.layernorm1',
                                                              'backbone.encoder.blocks.16.layernorm2',
                                                              'backbone.encoder.blocks.17.layernorm1',
                                                              'backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.10.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.10.output.mapping'],
            'backbone.encoder.blocks.10.output.mapping': ['backbone.encoder.blocks.10.layernorm2',
                                                          'backbone.encoder.blocks.10.output.projection'],
            'backbone.encoder.blocks.10.output.projection': ['backbone.encoder.blocks.10.output.mapping',
                                                             'backbone.encoder.blocks.10.output.dropout'],
            'backbone.encoder.blocks.10.output.dropout': ['backbone.encoder.blocks.10.output.projection',
                                                          ['backbone.encoder.blocks.11.layernorm1',
                                                           'backbone.encoder.blocks.11.layernorm2',
                                                           'backbone.encoder.blocks.12.layernorm1',
                                                           'backbone.encoder.blocks.12.layernorm2',
                                                           'backbone.encoder.blocks.13.layernorm1',
                                                           'backbone.encoder.blocks.13.layernorm2',
                                                           'backbone.encoder.blocks.14.layernorm1',
                                                           'backbone.encoder.blocks.14.layernorm2',
                                                           'backbone.encoder.blocks.15.layernorm1',
                                                           'backbone.encoder.blocks.15.layernorm2',
                                                           'backbone.encoder.blocks.16.layernorm1',
                                                           'backbone.encoder.blocks.16.layernorm2',
                                                           'backbone.encoder.blocks.17.layernorm1',
                                                           'backbone.encoder.blocks.17.layernorm2',
                                                           'backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.11.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.11.attention.dense1',
                                                       'backbone.encoder.blocks.11.attention.dense2',
                                                       'backbone.encoder.blocks.11.attention.dense3']],
            'backbone.encoder.blocks.11.attention.dense1': ['backbone.encoder.blocks.11.layernorm1',
                                                            'backbone.encoder.blocks.11.attention.softmax'],
            'backbone.encoder.blocks.11.attention.dense2': ['backbone.encoder.blocks.11.layernorm1',
                                                            'backbone.encoder.blocks.11.attention.softmax'],
            'backbone.encoder.blocks.11.attention.dense3': ['backbone.encoder.blocks.11.layernorm1',
                                                            'backbone.encoder.blocks.11.attention.projection'],

            'backbone.encoder.blocks.11.attention.softmax': [['backbone.encoder.blocks.11.attention.dense1',
                                                              'backbone.encoder.blocks.11.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.11.attention.prob_dropout'],

            'backbone.encoder.blocks.11.attention.prob_dropout': ['backbone.encoder.blocks.11.attention.softmax',
                                                                  'backbone.encoder.blocks.11.attention.projection'],

            'backbone.encoder.blocks.11.attention.projection': [['backbone.encoder.blocks.11.attention.dense3',
                                                                 'backbone.encoder.blocks.11.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.11.attention.dropout'],
            'backbone.encoder.blocks.11.attention.dropout': ['backbone.encoder.blocks.11.attention.projection',
                                                             ['backbone.encoder.blocks.11.layernorm2',
                                                              'backbone.encoder.blocks.12.layernorm1',
                                                              'backbone.encoder.blocks.12.layernorm2',
                                                              'backbone.encoder.blocks.13.layernorm1',
                                                              'backbone.encoder.blocks.13.layernorm2',
                                                              'backbone.encoder.blocks.14.layernorm1',
                                                              'backbone.encoder.blocks.14.layernorm2',
                                                              'backbone.encoder.blocks.15.layernorm1',
                                                              'backbone.encoder.blocks.15.layernorm2',
                                                              'backbone.encoder.blocks.16.layernorm1',
                                                              'backbone.encoder.blocks.16.layernorm2',
                                                              'backbone.encoder.blocks.17.layernorm1',
                                                              'backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.11.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.11.output.mapping'],
            'backbone.encoder.blocks.11.output.mapping': ['backbone.encoder.blocks.11.layernorm2',
                                                          'backbone.encoder.blocks.11.output.projection'],
            'backbone.encoder.blocks.11.output.projection': ['backbone.encoder.blocks.11.output.mapping',
                                                             'backbone.encoder.blocks.11.output.dropout'],
            'backbone.encoder.blocks.11.output.dropout': ['backbone.encoder.blocks.11.output.projection',
                                                          ['backbone.encoder.blocks.12.layernorm1',
                                                           'backbone.encoder.blocks.12.layernorm2',
                                                           'backbone.encoder.blocks.13.layernorm1',
                                                           'backbone.encoder.blocks.13.layernorm2',
                                                           'backbone.encoder.blocks.14.layernorm1',
                                                           'backbone.encoder.blocks.14.layernorm2',
                                                           'backbone.encoder.blocks.15.layernorm1',
                                                           'backbone.encoder.blocks.15.layernorm2',
                                                           'backbone.encoder.blocks.16.layernorm1',
                                                           'backbone.encoder.blocks.16.layernorm2',
                                                           'backbone.encoder.blocks.17.layernorm1',
                                                           'backbone.encoder.blocks.17.layernorm2',
                                                           'backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ###
            # ####
            'backbone.encoder.blocks.12.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.12.attention.dense1',
                                                       'backbone.encoder.blocks.12.attention.dense2',
                                                       'backbone.encoder.blocks.12.attention.dense3']],
            'backbone.encoder.blocks.12.attention.dense1': ['backbone.encoder.blocks.12.layernorm1',
                                                            'backbone.encoder.blocks.12.attention.softmax'],
            'backbone.encoder.blocks.12.attention.dense2': ['backbone.encoder.blocks.12.layernorm1',
                                                            'backbone.encoder.blocks.12.attention.softmax'],
            'backbone.encoder.blocks.12.attention.dense3': ['backbone.encoder.blocks.12.layernorm1',
                                                            'backbone.encoder.blocks.12.attention.projection'],

            'backbone.encoder.blocks.12.attention.softmax': [['backbone.encoder.blocks.12.attention.dense1',
                                                              'backbone.encoder.blocks.12.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.12.attention.prob_dropout'],

            'backbone.encoder.blocks.12.attention.prob_dropout': ['backbone.encoder.blocks.12.attention.softmax',
                                                                  'backbone.encoder.blocks.12.attention.projection'],

            'backbone.encoder.blocks.12.attention.projection': [['backbone.encoder.blocks.12.attention.dense3',
                                                                 'backbone.encoder.blocks.12.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.12.attention.dropout'],
            'backbone.encoder.blocks.12.attention.dropout': ['backbone.encoder.blocks.12.attention.projection',
                                                             ['backbone.encoder.blocks.12.layernorm2',
                                                              'backbone.encoder.blocks.13.layernorm1',
                                                              'backbone.encoder.blocks.13.layernorm2',
                                                              'backbone.encoder.blocks.14.layernorm1',
                                                              'backbone.encoder.blocks.14.layernorm2',
                                                              'backbone.encoder.blocks.15.layernorm1',
                                                              'backbone.encoder.blocks.15.layernorm2',
                                                              'backbone.encoder.blocks.16.layernorm1',
                                                              'backbone.encoder.blocks.16.layernorm2',
                                                              'backbone.encoder.blocks.17.layernorm1',
                                                              'backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.12.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.12.output.mapping'],
            'backbone.encoder.blocks.12.output.mapping': ['backbone.encoder.blocks.12.layernorm2',
                                                          'backbone.encoder.blocks.12.output.projection'],
            'backbone.encoder.blocks.12.output.projection': ['backbone.encoder.blocks.12.output.mapping',
                                                             'backbone.encoder.blocks.12.output.dropout'],
            'backbone.encoder.blocks.12.output.dropout': ['backbone.encoder.blocks.12.output.projection',
                                                          ['backbone.encoder.blocks.13.layernorm1',
                                                           'backbone.encoder.blocks.13.layernorm2',
                                                           'backbone.encoder.blocks.14.layernorm1',
                                                           'backbone.encoder.blocks.14.layernorm2',
                                                           'backbone.encoder.blocks.15.layernorm1',
                                                           'backbone.encoder.blocks.15.layernorm2',
                                                           'backbone.encoder.blocks.16.layernorm1',
                                                           'backbone.encoder.blocks.16.layernorm2',
                                                           'backbone.encoder.blocks.17.layernorm1',
                                                           'backbone.encoder.blocks.17.layernorm2',
                                                           'backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.13.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.13.attention.dense1',
                                                       'backbone.encoder.blocks.13.attention.dense2',
                                                       'backbone.encoder.blocks.13.attention.dense3']],
            'backbone.encoder.blocks.13.attention.dense1': ['backbone.encoder.blocks.13.layernorm1',
                                                            'backbone.encoder.blocks.13.attention.softmax'],
            'backbone.encoder.blocks.13.attention.dense2': ['backbone.encoder.blocks.13.layernorm1',
                                                            'backbone.encoder.blocks.13.attention.softmax'],
            'backbone.encoder.blocks.13.attention.dense3': ['backbone.encoder.blocks.13.layernorm1',
                                                            'backbone.encoder.blocks.13.attention.projection'],

            'backbone.encoder.blocks.13.attention.softmax': [['backbone.encoder.blocks.13.attention.dense1',
                                                              'backbone.encoder.blocks.13.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.13.attention.prob_dropout'],

            'backbone.encoder.blocks.13.attention.prob_dropout': ['backbone.encoder.blocks.13.attention.softmax',
                                                                  'backbone.encoder.blocks.13.attention.projection'],

            'backbone.encoder.blocks.13.attention.projection': [['backbone.encoder.blocks.13.attention.dense3',
                                                                 'backbone.encoder.blocks.13.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.13.attention.dropout'],
            'backbone.encoder.blocks.13.attention.dropout': ['backbone.encoder.blocks.13.attention.projection',
                                                             ['backbone.encoder.blocks.13.layernorm2',
                                                              'backbone.encoder.blocks.14.layernorm1',
                                                              'backbone.encoder.blocks.14.layernorm2',
                                                              'backbone.encoder.blocks.15.layernorm1',
                                                              'backbone.encoder.blocks.15.layernorm2',
                                                              'backbone.encoder.blocks.16.layernorm1',
                                                              'backbone.encoder.blocks.16.layernorm2',
                                                              'backbone.encoder.blocks.17.layernorm1',
                                                              'backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.13.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.13.output.mapping'],
            'backbone.encoder.blocks.13.output.mapping': ['backbone.encoder.blocks.13.layernorm2',
                                                          'backbone.encoder.blocks.13.output.projection'],
            'backbone.encoder.blocks.13.output.projection': ['backbone.encoder.blocks.13.output.mapping',
                                                             'backbone.encoder.blocks.13.output.dropout'],
            'backbone.encoder.blocks.13.output.dropout': ['backbone.encoder.blocks.13.output.projection',
                                                          ['backbone.encoder.blocks.14.layernorm1',
                                                           'backbone.encoder.blocks.14.layernorm2',
                                                           'backbone.encoder.blocks.15.layernorm1',
                                                           'backbone.encoder.blocks.15.layernorm2',
                                                           'backbone.encoder.blocks.16.layernorm1',
                                                           'backbone.encoder.blocks.16.layernorm2',
                                                           'backbone.encoder.blocks.17.layernorm1',
                                                           'backbone.encoder.blocks.17.layernorm2',
                                                           'backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.14.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.14.attention.dense1',
                                                       'backbone.encoder.blocks.14.attention.dense2',
                                                       'backbone.encoder.blocks.14.attention.dense3']],
            'backbone.encoder.blocks.14.attention.dense1': ['backbone.encoder.blocks.14.layernorm1',
                                                            'backbone.encoder.blocks.14.attention.softmax'],
            'backbone.encoder.blocks.14.attention.dense2': ['backbone.encoder.blocks.14.layernorm1',
                                                            'backbone.encoder.blocks.14.attention.softmax'],
            'backbone.encoder.blocks.14.attention.dense3': ['backbone.encoder.blocks.14.layernorm1',
                                                            'backbone.encoder.blocks.14.attention.projection'],

            'backbone.encoder.blocks.14.attention.softmax': [['backbone.encoder.blocks.14.attention.dense1',
                                                              'backbone.encoder.blocks.14.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.14.attention.prob_dropout'],

            'backbone.encoder.blocks.14.attention.prob_dropout': ['backbone.encoder.blocks.14.attention.softmax',
                                                                  'backbone.encoder.blocks.14.attention.projection'],

            'backbone.encoder.blocks.14.attention.projection': [['backbone.encoder.blocks.14.attention.dense3',
                                                                 'backbone.encoder.blocks.14.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.14.attention.dropout'],
            'backbone.encoder.blocks.14.attention.dropout': ['backbone.encoder.blocks.14.attention.projection',
                                                             ['backbone.encoder.blocks.14.layernorm2',
                                                              'backbone.encoder.blocks.15.layernorm1',
                                                              'backbone.encoder.blocks.15.layernorm2',
                                                              'backbone.encoder.blocks.16.layernorm1',
                                                              'backbone.encoder.blocks.16.layernorm2',
                                                              'backbone.encoder.blocks.17.layernorm1',
                                                              'backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.14.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.14.output.mapping'],
            'backbone.encoder.blocks.14.output.mapping': ['backbone.encoder.blocks.14.layernorm2',
                                                          'backbone.encoder.blocks.14.output.projection'],
            'backbone.encoder.blocks.14.output.projection': ['backbone.encoder.blocks.14.output.mapping',
                                                             'backbone.encoder.blocks.14.output.dropout'],
            'backbone.encoder.blocks.14.output.dropout': ['backbone.encoder.blocks.14.output.projection',
                                                          ['backbone.encoder.blocks.15.layernorm1',
                                                           'backbone.encoder.blocks.15.layernorm2',
                                                           'backbone.encoder.blocks.16.layernorm1',
                                                           'backbone.encoder.blocks.16.layernorm2',
                                                           'backbone.encoder.blocks.17.layernorm1',
                                                           'backbone.encoder.blocks.17.layernorm2',
                                                           'backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.15.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.15.attention.dense1',
                                                       'backbone.encoder.blocks.15.attention.dense2',
                                                       'backbone.encoder.blocks.15.attention.dense3']],
            'backbone.encoder.blocks.15.attention.dense1': ['backbone.encoder.blocks.15.layernorm1',
                                                            'backbone.encoder.blocks.15.attention.softmax'],
            'backbone.encoder.blocks.15.attention.dense2': ['backbone.encoder.blocks.15.layernorm1',
                                                            'backbone.encoder.blocks.15.attention.softmax'],
            'backbone.encoder.blocks.15.attention.dense3': ['backbone.encoder.blocks.15.layernorm1',
                                                            'backbone.encoder.blocks.15.attention.projection'],

            'backbone.encoder.blocks.15.attention.softmax': [['backbone.encoder.blocks.15.attention.dense1',
                                                              'backbone.encoder.blocks.15.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.15.attention.prob_dropout'],

            'backbone.encoder.blocks.15.attention.prob_dropout': ['backbone.encoder.blocks.15.attention.softmax',
                                                                  'backbone.encoder.blocks.15.attention.projection'],

            'backbone.encoder.blocks.15.attention.projection': [['backbone.encoder.blocks.15.attention.dense3',
                                                                 'backbone.encoder.blocks.15.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.15.attention.dropout'],
            'backbone.encoder.blocks.15.attention.dropout': ['backbone.encoder.blocks.15.attention.projection',
                                                             ['backbone.encoder.blocks.15.layernorm2',
                                                              'backbone.encoder.blocks.16.layernorm1',
                                                              'backbone.encoder.blocks.16.layernorm2',
                                                              'backbone.encoder.blocks.17.layernorm1',
                                                              'backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.15.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.15.output.mapping'],
            'backbone.encoder.blocks.15.output.mapping': ['backbone.encoder.blocks.15.layernorm2',
                                                          'backbone.encoder.blocks.15.output.projection'],
            'backbone.encoder.blocks.15.output.projection': ['backbone.encoder.blocks.15.output.mapping',
                                                             'backbone.encoder.blocks.15.output.dropout'],
            'backbone.encoder.blocks.15.output.dropout': ['backbone.encoder.blocks.15.output.projection',
                                                          ['backbone.encoder.blocks.16.layernorm1',
                                                           'backbone.encoder.blocks.16.layernorm2',
                                                           'backbone.encoder.blocks.17.layernorm1',
                                                           'backbone.encoder.blocks.17.layernorm2',
                                                           'backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.16.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.16.attention.dense1',
                                                       'backbone.encoder.blocks.16.attention.dense2',
                                                       'backbone.encoder.blocks.16.attention.dense3']],
            'backbone.encoder.blocks.16.attention.dense1': ['backbone.encoder.blocks.16.layernorm1',
                                                            'backbone.encoder.blocks.16.attention.softmax'],
            'backbone.encoder.blocks.16.attention.dense2': ['backbone.encoder.blocks.16.layernorm1',
                                                            'backbone.encoder.blocks.16.attention.softmax'],
            'backbone.encoder.blocks.16.attention.dense3': ['backbone.encoder.blocks.16.layernorm1',
                                                            'backbone.encoder.blocks.16.attention.projection'],

            'backbone.encoder.blocks.16.attention.softmax': [['backbone.encoder.blocks.16.attention.dense1',
                                                              'backbone.encoder.blocks.16.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.16.attention.prob_dropout'],

            'backbone.encoder.blocks.16.attention.prob_dropout': ['backbone.encoder.blocks.16.attention.softmax',
                                                                  'backbone.encoder.blocks.16.attention.projection'],

            'backbone.encoder.blocks.16.attention.projection': [['backbone.encoder.blocks.16.attention.dense3',
                                                                 'backbone.encoder.blocks.16.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.16.attention.dropout'],
            'backbone.encoder.blocks.16.attention.dropout': ['backbone.encoder.blocks.16.attention.projection',
                                                             ['backbone.encoder.blocks.16.layernorm2',
                                                              'backbone.encoder.blocks.17.layernorm1',
                                                              'backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.16.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.16.output.mapping'],
            'backbone.encoder.blocks.16.output.mapping': ['backbone.encoder.blocks.16.layernorm2',
                                                          'backbone.encoder.blocks.16.output.projection'],
            'backbone.encoder.blocks.16.output.projection': ['backbone.encoder.blocks.16.output.mapping',
                                                             'backbone.encoder.blocks.16.output.dropout'],
            'backbone.encoder.blocks.16.output.dropout': ['backbone.encoder.blocks.16.output.projection',
                                                          ['backbone.encoder.blocks.17.layernorm1',
                                                           'backbone.encoder.blocks.17.layernorm2',
                                                           'backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.17.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.17.attention.dense1',
                                                       'backbone.encoder.blocks.17.attention.dense2',
                                                       'backbone.encoder.blocks.17.attention.dense3']],
            'backbone.encoder.blocks.17.attention.dense1': ['backbone.encoder.blocks.17.layernorm1',
                                                            'backbone.encoder.blocks.17.attention.softmax'],
            'backbone.encoder.blocks.17.attention.dense2': ['backbone.encoder.blocks.17.layernorm1',
                                                            'backbone.encoder.blocks.17.attention.softmax'],
            'backbone.encoder.blocks.17.attention.dense3': ['backbone.encoder.blocks.17.layernorm1',
                                                            'backbone.encoder.blocks.17.attention.projection'],

            'backbone.encoder.blocks.17.attention.softmax': [['backbone.encoder.blocks.17.attention.dense1',
                                                              'backbone.encoder.blocks.17.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.17.attention.prob_dropout'],

            'backbone.encoder.blocks.17.attention.prob_dropout': ['backbone.encoder.blocks.17.attention.softmax',
                                                                  'backbone.encoder.blocks.17.attention.projection'],

            'backbone.encoder.blocks.17.attention.projection': [['backbone.encoder.blocks.17.attention.dense3',
                                                                 'backbone.encoder.blocks.17.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.17.attention.dropout'],
            'backbone.encoder.blocks.17.attention.dropout': ['backbone.encoder.blocks.17.attention.projection',
                                                             ['backbone.encoder.blocks.17.layernorm2',
                                                              'backbone.encoder.blocks.18.layernorm1',
                                                              'backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.17.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.17.output.mapping'],
            'backbone.encoder.blocks.17.output.mapping': ['backbone.encoder.blocks.17.layernorm2',
                                                          'backbone.encoder.blocks.17.output.projection'],
            'backbone.encoder.blocks.17.output.projection': ['backbone.encoder.blocks.17.output.mapping',
                                                             'backbone.encoder.blocks.17.output.dropout'],
            'backbone.encoder.blocks.17.output.dropout': ['backbone.encoder.blocks.17.output.projection',
                                                          ['backbone.encoder.blocks.18.layernorm1',
                                                           'backbone.encoder.blocks.18.layernorm2',
                                                           'backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.18.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.18.attention.dense1',
                                                       'backbone.encoder.blocks.18.attention.dense2',
                                                       'backbone.encoder.blocks.18.attention.dense3']],
            'backbone.encoder.blocks.18.attention.dense1': ['backbone.encoder.blocks.18.layernorm1',
                                                            'backbone.encoder.blocks.18.attention.softmax'],
            'backbone.encoder.blocks.18.attention.dense2': ['backbone.encoder.blocks.18.layernorm1',
                                                            'backbone.encoder.blocks.18.attention.softmax'],
            'backbone.encoder.blocks.18.attention.dense3': ['backbone.encoder.blocks.18.layernorm1',
                                                            'backbone.encoder.blocks.18.attention.projection'],

            'backbone.encoder.blocks.18.attention.softmax': [['backbone.encoder.blocks.18.attention.dense1',
                                                              'backbone.encoder.blocks.18.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.18.attention.prob_dropout'],

            'backbone.encoder.blocks.18.attention.prob_dropout': ['backbone.encoder.blocks.18.attention.softmax',
                                                                  'backbone.encoder.blocks.18.attention.projection'],

            'backbone.encoder.blocks.18.attention.projection': [['backbone.encoder.blocks.18.attention.dense3',
                                                                 'backbone.encoder.blocks.18.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.18.attention.dropout'],
            'backbone.encoder.blocks.18.attention.dropout': ['backbone.encoder.blocks.18.attention.projection',
                                                             ['backbone.encoder.blocks.18.layernorm2',
                                                              'backbone.encoder.blocks.19.layernorm1',
                                                              'backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.18.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.18.output.mapping'],
            'backbone.encoder.blocks.18.output.mapping': ['backbone.encoder.blocks.18.layernorm2',
                                                          'backbone.encoder.blocks.18.output.projection'],
            'backbone.encoder.blocks.18.output.projection': ['backbone.encoder.blocks.18.output.mapping',
                                                             'backbone.encoder.blocks.18.output.dropout'],
            'backbone.encoder.blocks.18.output.dropout': ['backbone.encoder.blocks.18.output.projection',
                                                          ['backbone.encoder.blocks.19.layernorm1',
                                                           'backbone.encoder.blocks.19.layernorm2',
                                                           'backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.19.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.19.attention.dense1',
                                                       'backbone.encoder.blocks.19.attention.dense2',
                                                       'backbone.encoder.blocks.19.attention.dense3']],
            'backbone.encoder.blocks.19.attention.dense1': ['backbone.encoder.blocks.19.layernorm1',
                                                            'backbone.encoder.blocks.19.attention.softmax'],
            'backbone.encoder.blocks.19.attention.dense2': ['backbone.encoder.blocks.19.layernorm1',
                                                            'backbone.encoder.blocks.19.attention.softmax'],
            'backbone.encoder.blocks.19.attention.dense3': ['backbone.encoder.blocks.19.layernorm1',
                                                            'backbone.encoder.blocks.19.attention.projection'],

            'backbone.encoder.blocks.19.attention.softmax': [['backbone.encoder.blocks.19.attention.dense1',
                                                              'backbone.encoder.blocks.19.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.19.attention.prob_dropout'],

            'backbone.encoder.blocks.19.attention.prob_dropout': ['backbone.encoder.blocks.19.attention.softmax',
                                                                  'backbone.encoder.blocks.19.attention.projection'],

            'backbone.encoder.blocks.19.attention.projection': [['backbone.encoder.blocks.19.attention.dense3',
                                                                 'backbone.encoder.blocks.19.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.19.attention.dropout'],
            'backbone.encoder.blocks.19.attention.dropout': ['backbone.encoder.blocks.19.attention.projection',
                                                             ['backbone.encoder.blocks.19.layernorm2',
                                                              'backbone.encoder.blocks.20.layernorm1',
                                                              'backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.19.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.19.output.mapping'],
            'backbone.encoder.blocks.19.output.mapping': ['backbone.encoder.blocks.19.layernorm2',
                                                          'backbone.encoder.blocks.19.output.projection'],
            'backbone.encoder.blocks.19.output.projection': ['backbone.encoder.blocks.19.output.mapping',
                                                             'backbone.encoder.blocks.19.output.dropout'],
            'backbone.encoder.blocks.19.output.dropout': ['backbone.encoder.blocks.19.output.projection',
                                                          ['backbone.encoder.blocks.20.layernorm1',
                                                           'backbone.encoder.blocks.20.layernorm2',
                                                           'backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.20.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.20.attention.dense1',
                                                       'backbone.encoder.blocks.20.attention.dense2',
                                                       'backbone.encoder.blocks.20.attention.dense3']],
            'backbone.encoder.blocks.20.attention.dense1': ['backbone.encoder.blocks.20.layernorm1',
                                                            'backbone.encoder.blocks.20.attention.softmax'],
            'backbone.encoder.blocks.20.attention.dense2': ['backbone.encoder.blocks.20.layernorm1',
                                                            'backbone.encoder.blocks.20.attention.softmax'],
            'backbone.encoder.blocks.20.attention.dense3': ['backbone.encoder.blocks.20.layernorm1',
                                                            'backbone.encoder.blocks.20.attention.projection'],

            'backbone.encoder.blocks.20.attention.softmax': [['backbone.encoder.blocks.20.attention.dense1',
                                                              'backbone.encoder.blocks.20.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.20.attention.prob_dropout'],

            'backbone.encoder.blocks.20.attention.prob_dropout': ['backbone.encoder.blocks.20.attention.softmax',
                                                                  'backbone.encoder.blocks.20.attention.projection'],

            'backbone.encoder.blocks.20.attention.projection': [['backbone.encoder.blocks.20.attention.dense3',
                                                                 'backbone.encoder.blocks.20.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.20.attention.dropout'],
            'backbone.encoder.blocks.20.attention.dropout': ['backbone.encoder.blocks.20.attention.projection',
                                                             ['backbone.encoder.blocks.20.layernorm2',
                                                              'backbone.encoder.blocks.21.layernorm1',
                                                              'backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm'
                                                              ]],

            'backbone.encoder.blocks.20.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       'backbone.encoder.blocks.20.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.20.output.mapping'],
            'backbone.encoder.blocks.20.output.mapping': ['backbone.encoder.blocks.20.layernorm2',
                                                          'backbone.encoder.blocks.20.output.projection'],
            'backbone.encoder.blocks.20.output.projection': ['backbone.encoder.blocks.20.output.mapping',
                                                             'backbone.encoder.blocks.20.output.dropout'],
            'backbone.encoder.blocks.20.output.dropout': ['backbone.encoder.blocks.20.output.projection',
                                                          ['backbone.encoder.blocks.21.layernorm1',
                                                           'backbone.encoder.blocks.21.layernorm2',
                                                           'backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.21.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       'backbone.encoder.blocks.20.attention.dropout',
                                                       'backbone.encoder.blocks.20.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.21.attention.dense1',
                                                       'backbone.encoder.blocks.21.attention.dense2',
                                                       'backbone.encoder.blocks.21.attention.dense3']],
            'backbone.encoder.blocks.21.attention.dense1': ['backbone.encoder.blocks.21.layernorm1',
                                                            'backbone.encoder.blocks.21.attention.softmax'],
            'backbone.encoder.blocks.21.attention.dense2': ['backbone.encoder.blocks.21.layernorm1',
                                                            'backbone.encoder.blocks.21.attention.softmax'],
            'backbone.encoder.blocks.21.attention.dense3': ['backbone.encoder.blocks.21.layernorm1',
                                                            'backbone.encoder.blocks.21.attention.projection'],

            'backbone.encoder.blocks.21.attention.softmax': [['backbone.encoder.blocks.21.attention.dense1',
                                                              'backbone.encoder.blocks.21.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.21.attention.prob_dropout'],

            'backbone.encoder.blocks.21.attention.prob_dropout': ['backbone.encoder.blocks.21.attention.softmax',
                                                                  'backbone.encoder.blocks.21.attention.projection'],

            'backbone.encoder.blocks.21.attention.projection': [['backbone.encoder.blocks.21.attention.dense3',
                                                                 'backbone.encoder.blocks.21.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.21.attention.dropout'],
            'backbone.encoder.blocks.21.attention.dropout': ['backbone.encoder.blocks.21.attention.projection',
                                                             ['backbone.encoder.blocks.21.layernorm2',
                                                              'backbone.encoder.blocks.22.layernorm1',
                                                              'backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.21.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       'backbone.encoder.blocks.20.attention.dropout',
                                                       'backbone.encoder.blocks.20.output.dropout',
                                                       'backbone.encoder.blocks.21.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.21.output.mapping'],
            'backbone.encoder.blocks.21.output.mapping': ['backbone.encoder.blocks.21.layernorm2',
                                                          'backbone.encoder.blocks.21.output.projection'],
            'backbone.encoder.blocks.21.output.projection': ['backbone.encoder.blocks.21.output.mapping',
                                                             'backbone.encoder.blocks.21.output.dropout'],
            'backbone.encoder.blocks.21.output.dropout': ['backbone.encoder.blocks.21.output.projection',
                                                          ['backbone.encoder.blocks.22.layernorm1',
                                                           'backbone.encoder.blocks.22.layernorm2',
                                                           'backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.22.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       'backbone.encoder.blocks.20.attention.dropout',
                                                       'backbone.encoder.blocks.20.output.dropout',
                                                       'backbone.encoder.blocks.21.attention.dropout',
                                                       'backbone.encoder.blocks.21.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.22.attention.dense1',
                                                       'backbone.encoder.blocks.22.attention.dense2',
                                                       'backbone.encoder.blocks.22.attention.dense3']],
            'backbone.encoder.blocks.22.attention.dense1': ['backbone.encoder.blocks.22.layernorm1',
                                                            'backbone.encoder.blocks.22.attention.softmax'],
            'backbone.encoder.blocks.22.attention.dense2': ['backbone.encoder.blocks.22.layernorm1',
                                                            'backbone.encoder.blocks.22.attention.softmax'],
            'backbone.encoder.blocks.22.attention.dense3': ['backbone.encoder.blocks.22.layernorm1',
                                                            'backbone.encoder.blocks.22.attention.projection'],

            'backbone.encoder.blocks.22.attention.softmax': [['backbone.encoder.blocks.22.attention.dense1',
                                                              'backbone.encoder.blocks.22.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.22.attention.prob_dropout'],

            'backbone.encoder.blocks.22.attention.prob_dropout': ['backbone.encoder.blocks.22.attention.softmax',
                                                                  'backbone.encoder.blocks.22.attention.projection'],

            'backbone.encoder.blocks.22.attention.projection': [['backbone.encoder.blocks.22.attention.dense3',
                                                                 'backbone.encoder.blocks.22.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.22.attention.dropout'],
            'backbone.encoder.blocks.22.attention.dropout': ['backbone.encoder.blocks.22.attention.projection',
                                                             ['backbone.encoder.blocks.22.layernorm2',
                                                              'backbone.encoder.blocks.23.layernorm1',
                                                              'backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.22.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       'backbone.encoder.blocks.20.attention.dropout',
                                                       'backbone.encoder.blocks.20.output.dropout',
                                                       'backbone.encoder.blocks.21.attention.dropout',
                                                       'backbone.encoder.blocks.21.output.dropout',
                                                       'backbone.encoder.blocks.22.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.22.output.mapping'],
            'backbone.encoder.blocks.22.output.mapping': ['backbone.encoder.blocks.22.layernorm2',
                                                          'backbone.encoder.blocks.22.output.projection'],
            'backbone.encoder.blocks.22.output.projection': ['backbone.encoder.blocks.22.output.mapping',
                                                             'backbone.encoder.blocks.22.output.dropout'],
            'backbone.encoder.blocks.22.output.dropout': ['backbone.encoder.blocks.22.output.projection',
                                                          ['backbone.encoder.blocks.23.layernorm1',
                                                           'backbone.encoder.blocks.23.layernorm2',
                                                           'backbone.layernorm']],

            # ####
            'backbone.encoder.blocks.23.layernorm1': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       'backbone.encoder.blocks.20.attention.dropout',
                                                       'backbone.encoder.blocks.20.output.dropout',
                                                       'backbone.encoder.blocks.21.attention.dropout',
                                                       'backbone.encoder.blocks.21.output.dropout',
                                                       'backbone.encoder.blocks.22.attention.dropout',
                                                       'backbone.encoder.blocks.22.output.dropout',
                                                       ],
                                                      ['backbone.encoder.blocks.23.attention.dense1',
                                                       'backbone.encoder.blocks.23.attention.dense2',
                                                       'backbone.encoder.blocks.23.attention.dense3']],
            'backbone.encoder.blocks.23.attention.dense1': ['backbone.encoder.blocks.23.layernorm1',
                                                            'backbone.encoder.blocks.23.attention.softmax'],
            'backbone.encoder.blocks.23.attention.dense2': ['backbone.encoder.blocks.23.layernorm1',
                                                            'backbone.encoder.blocks.23.attention.softmax'],
            'backbone.encoder.blocks.23.attention.dense3': ['backbone.encoder.blocks.23.layernorm1',
                                                            'backbone.encoder.blocks.23.attention.projection'],

            'backbone.encoder.blocks.23.attention.softmax': [['backbone.encoder.blocks.23.attention.dense1',
                                                              'backbone.encoder.blocks.23.attention.dense2',
                                                              'backbone.get_attention_mask'],
                                                             'backbone.encoder.blocks.23.attention.prob_dropout'],

            'backbone.encoder.blocks.23.attention.prob_dropout': ['backbone.encoder.blocks.23.attention.softmax',
                                                                  'backbone.encoder.blocks.23.attention.projection'],

            'backbone.encoder.blocks.23.attention.projection': [['backbone.encoder.blocks.23.attention.dense3',
                                                                 'backbone.encoder.blocks.23.attention.prob_dropout'],
                                                                'backbone.encoder.blocks.23.attention.dropout'],
            'backbone.encoder.blocks.23.attention.dropout': ['backbone.encoder.blocks.23.attention.projection',
                                                             ['backbone.encoder.blocks.23.layernorm2',
                                                              'backbone.layernorm']],

            'backbone.encoder.blocks.23.layernorm2': [['backbone.word_embedding', 'backbone.position_embedding',
                                                       'backbone.encoder.blocks.0.attention.dropout',
                                                       'backbone.encoder.blocks.0.output.dropout',
                                                       'backbone.encoder.blocks.1.attention.dropout',
                                                       'backbone.encoder.blocks.1.output.dropout',
                                                       'backbone.encoder.blocks.2.attention.dropout',
                                                       'backbone.encoder.blocks.2.output.dropout',
                                                       'backbone.encoder.blocks.3.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.3.output.dropout',
                                                       'backbone.encoder.blocks.4.attention.dropout',
                                                       'backbone.encoder.blocks.4.output.dropout',
                                                       'backbone.encoder.blocks.5.attention.dropout',
                                                       'backbone.encoder.blocks.5.output.dropout',
                                                       'backbone.encoder.blocks.6.attention.dropout',
                                                       'backbone.encoder.blocks.6.output.dropout',
                                                       'backbone.encoder.blocks.7.attention.dropout',
                                                       'backbone.encoder.blocks.7.output.dropout',
                                                       'backbone.encoder.blocks.8.attention.dropout',
                                                       'backbone.encoder.blocks.8.output.dropout',
                                                       'backbone.encoder.blocks.9.attention.dropout',
                                                       'backbone.encoder.blocks.9.output.dropout',
                                                       'backbone.encoder.blocks.10.attention.dropout',
                                                       'backbone.encoder.blocks.10.output.dropout',
                                                       'backbone.encoder.blocks.11.attention.dropout',
                                                       'backbone.encoder.blocks.11.output.dropout',
                                                       'backbone.encoder.blocks.12.attention.dropout',
                                                       'backbone.encoder.blocks.12.output.dropout',
                                                       'backbone.encoder.blocks.13.attention.dropout',
                                                       'backbone.encoder.blocks.13.output.dropout',
                                                       'backbone.encoder.blocks.14.attention.dropout',
                                                       'backbone.encoder.blocks.14.output.dropout',
                                                       'backbone.encoder.blocks.15.attention.dropout',
                                                       'backbone.encoder.blocks.15.output.dropout',
                                                       'backbone.encoder.blocks.16.attention.dropout',
                                                       'backbone.encoder.blocks.16.output.dropout',
                                                       'backbone.encoder.blocks.17.attention.dropout',
                                                       'backbone.encoder.blocks.17.output.dropout',
                                                       'backbone.encoder.blocks.18.attention.dropout',
                                                       'backbone.encoder.blocks.18.output.dropout',
                                                       'backbone.encoder.blocks.19.attention.dropout',
                                                       'backbone.encoder.blocks.19.output.dropout',
                                                       'backbone.encoder.blocks.20.attention.dropout',
                                                       'backbone.encoder.blocks.20.output.dropout',
                                                       'backbone.encoder.blocks.21.attention.dropout',
                                                       'backbone.encoder.blocks.21.output.dropout',
                                                       'backbone.encoder.blocks.22.attention.dropout',
                                                       'backbone.encoder.blocks.22.output.dropout',
                                                       'backbone.encoder.blocks.23.attention.dropout',
                                                       ],
                                                      'backbone.encoder.blocks.23.output.mapping'],
            'backbone.encoder.blocks.23.output.mapping': ['backbone.encoder.blocks.23.layernorm2',
                                                          'backbone.encoder.blocks.23.output.projection'],
            'backbone.encoder.blocks.23.output.projection': ['backbone.encoder.blocks.23.output.mapping',
                                                             'backbone.encoder.blocks.23.output.dropout'],
            'backbone.encoder.blocks.23.output.dropout': ['backbone.encoder.blocks.23.output.projection',
                                                          'backbone.layernorm'],

            # ###
            'backbone.layernorm': [['backbone.word_embedding', 'backbone.position_embedding',
                                    'backbone.encoder.blocks.0.attention.dropout',
                                    'backbone.encoder.blocks.0.output.dropout',
                                    'backbone.encoder.blocks.1.attention.dropout',
                                    'backbone.encoder.blocks.1.output.dropout',
                                    'backbone.encoder.blocks.2.attention.dropout',
                                    'backbone.encoder.blocks.2.output.dropout',
                                    'backbone.encoder.blocks.3.attention.dropout',
                                    'backbone.encoder.blocks.3.output.dropout',
                                    'backbone.encoder.blocks.4.attention.dropout',
                                    'backbone.encoder.blocks.4.output.dropout',
                                    'backbone.encoder.blocks.5.attention.dropout',
                                    'backbone.encoder.blocks.5.output.dropout',
                                    'backbone.encoder.blocks.6.attention.dropout',
                                    'backbone.encoder.blocks.6.output.dropout',
                                    'backbone.encoder.blocks.7.attention.dropout',
                                    'backbone.encoder.blocks.7.output.dropout',
                                    'backbone.encoder.blocks.8.attention.dropout',
                                    'backbone.encoder.blocks.8.output.dropout',
                                    'backbone.encoder.blocks.9.attention.dropout',
                                    'backbone.encoder.blocks.9.output.dropout',
                                    'backbone.encoder.blocks.10.attention.dropout',
                                    'backbone.encoder.blocks.10.output.dropout',
                                    'backbone.encoder.blocks.11.attention.dropout',
                                    'backbone.encoder.blocks.11.output.dropout',
                                    'backbone.encoder.blocks.12.attention.dropout',
                                    'backbone.encoder.blocks.12.output.dropout',
                                    'backbone.encoder.blocks.13.attention.dropout',
                                    'backbone.encoder.blocks.13.output.dropout',
                                    'backbone.encoder.blocks.14.attention.dropout',
                                    'backbone.encoder.blocks.14.output.dropout',
                                    'backbone.encoder.blocks.15.attention.dropout',
                                    'backbone.encoder.blocks.15.output.dropout',
                                    'backbone.encoder.blocks.16.attention.dropout',
                                    'backbone.encoder.blocks.16.output.dropout',
                                    'backbone.encoder.blocks.17.attention.dropout',
                                    'backbone.encoder.blocks.17.output.dropout',
                                    'backbone.encoder.blocks.18.attention.dropout',
                                    'backbone.encoder.blocks.18.output.dropout',
                                    'backbone.encoder.blocks.19.attention.dropout',
                                    'backbone.encoder.blocks.19.output.dropout',
                                    'backbone.encoder.blocks.20.attention.dropout',
                                    'backbone.encoder.blocks.20.output.dropout',
                                    'backbone.encoder.blocks.21.attention.dropout',
                                    'backbone.encoder.blocks.21.output.dropout',
                                    'backbone.encoder.blocks.22.attention.dropout',
                                    'backbone.encoder.blocks.22.output.dropout',
                                    'backbone.encoder.blocks.23.attention.dropout',
                                    'backbone.encoder.blocks.23.output.dropout',
                                    ],
                                   'head'],

            'head': [['backbone.layernorm', 'backbone.word_embedding'], 'OUTPUT']
        }

        self.layer_input_dtype = {
            'backbone.get_attention_mask': [torch.float32],
            'backbone.word_embedding': [torch.int32],
            'backbone.position_embedding': [torch.int64],
            'backbone.encoder.blocks.0.layernorm1': [torch.float32],
            'backbone.encoder.blocks.0.layernorm2': [torch.float32],
            'backbone.encoder.blocks.0.attention.projection': [torch.float32],
            'backbone.encoder.blocks.0.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.0.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.0.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.0.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.0.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.0.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.0.output.hidden_act': [torch.float32],
            'backbone.encoder.blocks.0.output.mapping': [torch.float32],
            'backbone.encoder.blocks.0.output.projection': [torch.float32],
            'backbone.encoder.blocks.0.output.dropout': [torch.float32],
            'backbone.encoder.blocks.1.layernorm1': [torch.float32],
            'backbone.encoder.blocks.1.layernorm2': [torch.float32],
            'backbone.encoder.blocks.1.attention.projection': [torch.float32],
            'backbone.encoder.blocks.1.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.1.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.1.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.1.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.1.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.1.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.1.output.mapping': [torch.float32],
            'backbone.encoder.blocks.1.output.projection': [torch.float32],
            'backbone.encoder.blocks.1.output.dropout': [torch.float32],
            'backbone.encoder.blocks.2.layernorm1': [torch.float32],
            'backbone.encoder.blocks.2.layernorm2': [torch.float32],
            'backbone.encoder.blocks.2.attention.projection': [torch.float32],
            'backbone.encoder.blocks.2.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.2.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.2.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.2.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.2.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.2.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.2.output.mapping': [torch.float32],
            'backbone.encoder.blocks.2.output.projection': [torch.float32],
            'backbone.encoder.blocks.2.output.dropout': [torch.float32],
            'backbone.encoder.blocks.3.layernorm1': [torch.float32],
            'backbone.encoder.blocks.3.layernorm2': [torch.float32],
            'backbone.encoder.blocks.3.attention.projection': [torch.float32],
            'backbone.encoder.blocks.3.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.3.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.3.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.3.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.3.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.3.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.3.output.mapping': [torch.float32],
            'backbone.encoder.blocks.3.output.projection': [torch.float32],
            'backbone.encoder.blocks.3.output.dropout': [torch.float32],
            'backbone.encoder.blocks.4.layernorm1': [torch.float32],
            'backbone.encoder.blocks.4.layernorm2': [torch.float32],
            'backbone.encoder.blocks.4.attention.projection': [torch.float32],
            'backbone.encoder.blocks.4.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.4.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.4.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.4.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.4.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.4.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.4.output.mapping': [torch.float32],
            'backbone.encoder.blocks.4.output.projection': [torch.float32],
            'backbone.encoder.blocks.4.output.dropout': [torch.float32],
            'backbone.encoder.blocks.5.layernorm1': [torch.float32],
            'backbone.encoder.blocks.5.layernorm2': [torch.float32],
            'backbone.encoder.blocks.5.attention.projection': [torch.float32],
            'backbone.encoder.blocks.5.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.5.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.5.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.5.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.5.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.5.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.5.output.mapping': [torch.float32],
            'backbone.encoder.blocks.5.output.projection': [torch.float32],
            'backbone.encoder.blocks.5.output.dropout': [torch.float32],
            'backbone.encoder.blocks.6.layernorm1': [torch.float32],
            'backbone.encoder.blocks.6.layernorm2': [torch.float32],
            'backbone.encoder.blocks.6.attention.projection': [torch.float32],
            'backbone.encoder.blocks.6.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.6.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.6.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.6.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.6.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.6.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.6.output.mapping': [torch.float32],
            'backbone.encoder.blocks.6.output.projection': [torch.float32],
            'backbone.encoder.blocks.6.output.dropout': [torch.float32],
            'backbone.encoder.blocks.7.layernorm1': [torch.float32],
            'backbone.encoder.blocks.7.layernorm2': [torch.float32],
            'backbone.encoder.blocks.7.attention.projection': [torch.float32],
            'backbone.encoder.blocks.7.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.7.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.7.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.7.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.7.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.7.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.7.output.mapping': [torch.float32],
            'backbone.encoder.blocks.7.output.projection': [torch.float32],
            'backbone.encoder.blocks.7.output.dropout': [torch.float32],
            'backbone.encoder.blocks.8.layernorm1': [torch.float32],
            'backbone.encoder.blocks.8.layernorm2': [torch.float32],
            'backbone.encoder.blocks.8.attention.projection': [torch.float32],
            'backbone.encoder.blocks.8.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.8.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.8.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.8.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.8.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.8.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.8.output.mapping': [torch.float32],
            'backbone.encoder.blocks.8.output.projection': [torch.float32],
            'backbone.encoder.blocks.8.output.dropout': [torch.float32],
            'backbone.encoder.blocks.9.layernorm1': [torch.float32],
            'backbone.encoder.blocks.9.layernorm2': [torch.float32],
            'backbone.encoder.blocks.9.attention.projection': [torch.float32],
            'backbone.encoder.blocks.9.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.9.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.9.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.9.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.9.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.9.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.9.output.mapping': [torch.float32],
            'backbone.encoder.blocks.9.output.projection': [torch.float32],
            'backbone.encoder.blocks.9.output.dropout': [torch.float32],
            'backbone.encoder.blocks.10.layernorm1': [torch.float32],
            'backbone.encoder.blocks.10.layernorm2': [torch.float32],
            'backbone.encoder.blocks.10.attention.projection': [torch.float32],
            'backbone.encoder.blocks.10.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.10.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.10.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.10.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.10.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.10.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.10.output.mapping': [torch.float32],
            'backbone.encoder.blocks.10.output.projection': [torch.float32],
            'backbone.encoder.blocks.10.output.dropout': [torch.float32],
            'backbone.encoder.blocks.11.layernorm1': [torch.float32],
            'backbone.encoder.blocks.11.layernorm2': [torch.float32],
            'backbone.encoder.blocks.11.attention.projection': [torch.float32],
            'backbone.encoder.blocks.11.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.11.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.11.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.11.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.11.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.11.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.11.output.mapping': [torch.float32],
            'backbone.encoder.blocks.11.output.projection': [torch.float32],
            'backbone.encoder.blocks.11.output.dropout': [torch.float32],
            'backbone.encoder.blocks.12.layernorm1': [torch.float32],
            'backbone.encoder.blocks.12.layernorm2': [torch.float32],
            'backbone.encoder.blocks.12.attention.projection': [torch.float32],
            'backbone.encoder.blocks.12.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.12.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.12.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.12.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.12.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.12.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.12.output.mapping': [torch.float32],
            'backbone.encoder.blocks.12.output.projection': [torch.float32],
            'backbone.encoder.blocks.12.output.dropout': [torch.float32],
            'backbone.encoder.blocks.13.layernorm1': [torch.float32],
            'backbone.encoder.blocks.13.layernorm2': [torch.float32],
            'backbone.encoder.blocks.13.attention.projection': [torch.float32],
            'backbone.encoder.blocks.13.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.13.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.13.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.13.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.13.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.13.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.13.output.mapping': [torch.float32],
            'backbone.encoder.blocks.13.output.projection': [torch.float32],
            'backbone.encoder.blocks.13.output.dropout': [torch.float32],
            'backbone.encoder.blocks.14.layernorm1': [torch.float32],
            'backbone.encoder.blocks.14.layernorm2': [torch.float32],
            'backbone.encoder.blocks.14.attention.projection': [torch.float32],
            'backbone.encoder.blocks.14.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.14.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.14.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.14.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.14.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.14.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.14.output.mapping': [torch.float32],
            'backbone.encoder.blocks.14.output.projection': [torch.float32],
            'backbone.encoder.blocks.14.output.dropout': [torch.float32],
            'backbone.encoder.blocks.15.layernorm1': [torch.float32],
            'backbone.encoder.blocks.15.layernorm2': [torch.float32],
            'backbone.encoder.blocks.15.attention.projection': [torch.float32],
            'backbone.encoder.blocks.15.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.15.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.15.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.15.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.15.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.15.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.15.output.mapping': [torch.float32],
            'backbone.encoder.blocks.15.output.projection': [torch.float32],
            'backbone.encoder.blocks.15.output.dropout': [torch.float32],
            'backbone.encoder.blocks.16.layernorm1': [torch.float32],
            'backbone.encoder.blocks.16.layernorm2': [torch.float32],
            'backbone.encoder.blocks.16.attention.projection': [torch.float32],
            'backbone.encoder.blocks.16.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.16.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.16.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.16.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.16.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.16.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.16.output.mapping': [torch.float32],
            'backbone.encoder.blocks.16.output.projection': [torch.float32],
            'backbone.encoder.blocks.16.output.dropout': [torch.float32],
            'backbone.encoder.blocks.17.layernorm1': [torch.float32],
            'backbone.encoder.blocks.17.layernorm2': [torch.float32],
            'backbone.encoder.blocks.17.attention.projection': [torch.float32],
            'backbone.encoder.blocks.17.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.17.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.17.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.17.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.17.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.17.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.17.output.mapping': [torch.float32],
            'backbone.encoder.blocks.17.output.projection': [torch.float32],
            'backbone.encoder.blocks.17.output.dropout': [torch.float32],
            'backbone.encoder.blocks.18.layernorm1': [torch.float32],
            'backbone.encoder.blocks.18.layernorm2': [torch.float32],
            'backbone.encoder.blocks.18.attention.projection': [torch.float32],
            'backbone.encoder.blocks.18.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.18.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.18.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.18.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.18.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.18.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.18.output.mapping': [torch.float32],
            'backbone.encoder.blocks.18.output.projection': [torch.float32],
            'backbone.encoder.blocks.18.output.dropout': [torch.float32],
            'backbone.encoder.blocks.19.layernorm1': [torch.float32],
            'backbone.encoder.blocks.19.layernorm2': [torch.float32],
            'backbone.encoder.blocks.19.attention.projection': [torch.float32],
            'backbone.encoder.blocks.19.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.19.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.19.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.19.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.19.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.19.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.19.output.mapping': [torch.float32],
            'backbone.encoder.blocks.19.output.projection': [torch.float32],
            'backbone.encoder.blocks.19.output.dropout': [torch.float32],
            'backbone.encoder.blocks.20.layernorm1': [torch.float32],
            'backbone.encoder.blocks.20.layernorm2': [torch.float32],
            'backbone.encoder.blocks.20.attention.projection': [torch.float32],
            'backbone.encoder.blocks.20.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.20.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.20.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.20.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.20.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.20.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.20.output.mapping': [torch.float32],
            'backbone.encoder.blocks.20.output.projection': [torch.float32],
            'backbone.encoder.blocks.20.output.dropout': [torch.float32],
            'backbone.encoder.blocks.21.layernorm1': [torch.float32],
            'backbone.encoder.blocks.21.layernorm2': [torch.float32],
            'backbone.encoder.blocks.21.attention.projection': [torch.float32],
            'backbone.encoder.blocks.21.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.21.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.21.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.21.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.21.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.21.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.21.output.mapping': [torch.float32],
            'backbone.encoder.blocks.21.output.projection': [torch.float32],
            'backbone.encoder.blocks.21.output.dropout': [torch.float32],
            'backbone.encoder.blocks.22.layernorm1': [torch.float32],
            'backbone.encoder.blocks.22.layernorm2': [torch.float32],
            'backbone.encoder.blocks.22.attention.projection': [torch.float32],
            'backbone.encoder.blocks.22.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.22.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.22.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.22.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.22.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.22.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.22.output.mapping': [torch.float32],
            'backbone.encoder.blocks.22.output.projection': [torch.float32],
            'backbone.encoder.blocks.22.output.dropout': [torch.float32],
            'backbone.encoder.blocks.23.layernorm1': [torch.float32],
            'backbone.encoder.blocks.23.layernorm2': [torch.float32],
            'backbone.encoder.blocks.23.attention.projection': [torch.float32],
            'backbone.encoder.blocks.23.attention.dropout': [torch.float32],
            'backbone.encoder.blocks.23.attention.prob_dropout': [torch.float32],
            'backbone.encoder.blocks.23.attention.softmax': [torch.float32],
            'backbone.encoder.blocks.23.attention.dense1': [torch.float32],
            'backbone.encoder.blocks.23.attention.dense2': [torch.float32],
            'backbone.encoder.blocks.23.attention.dense3': [torch.float32],
            'backbone.encoder.blocks.23.output.mapping': [torch.float32],
            'backbone.encoder.blocks.23.output.projection': [torch.float32],
            'backbone.encoder.blocks.23.output.dropout': [torch.float32],
            'backbone.layernorm': [torch.float32],
            'head': [torch.float32, torch.float32]
        }

        self.layer_names = {
            "backbone": self.backbone,
            "backbone.get_attention_mask": self.backbone.get_attention_mask,
            "backbone.word_embedding": self.backbone.word_embedding,
            "backbone.position_embedding": self.backbone.position_embedding,
            "backbone.encoder": self.backbone.encoder,
            "backbone.encoder.blocks": self.backbone.encoder.blocks,
            "backbone.encoder.blocks.0": self.backbone.encoder.blocks[0],
            "backbone.encoder.blocks.0.layernorm1": self.backbone.encoder.blocks[0].layernorm1,
            "backbone.encoder.blocks.0.layernorm2": self.backbone.encoder.blocks[0].layernorm2,
            "backbone.encoder.blocks.0.attention": self.backbone.encoder.blocks[0].attention,
            "backbone.encoder.blocks.0.attention.projection": self.backbone.encoder.blocks[0].attention.projection,
            "backbone.encoder.blocks.0.attention.dropout": self.backbone.encoder.blocks[0].attention.dropout,
            "backbone.encoder.blocks.0.attention.prob_dropout": self.backbone.encoder.blocks[0].attention.prob_dropout,
            "backbone.encoder.blocks.0.attention.softmax": self.backbone.encoder.blocks[0].attention.softmax,
            "backbone.encoder.blocks.0.attention.dense1": self.backbone.encoder.blocks[0].attention.dense1,
            "backbone.encoder.blocks.0.attention.dense2": self.backbone.encoder.blocks[0].attention.dense2,
            "backbone.encoder.blocks.0.attention.dense3": self.backbone.encoder.blocks[0].attention.dense3,
            "backbone.encoder.blocks.0.output": self.backbone.encoder.blocks[0].output,
            "backbone.encoder.blocks.0.output.hidden_act": self.backbone.encoder.blocks[0].output.hidden_act,
            "backbone.encoder.blocks.0.output.mapping": self.backbone.encoder.blocks[0].output.mapping,
            "backbone.encoder.blocks.0.output.projection": self.backbone.encoder.blocks[0].output.projection,
            "backbone.encoder.blocks.0.output.dropout": self.backbone.encoder.blocks[0].output.dropout,
            "backbone.encoder.blocks.1": self.backbone.encoder.blocks[1],
            "backbone.encoder.blocks.1.layernorm1": self.backbone.encoder.blocks[1].layernorm1,
            "backbone.encoder.blocks.1.layernorm2": self.backbone.encoder.blocks[1].layernorm2,
            "backbone.encoder.blocks.1.attention": self.backbone.encoder.blocks[1].attention,
            "backbone.encoder.blocks.1.attention.projection": self.backbone.encoder.blocks[1].attention.projection,
            "backbone.encoder.blocks.1.attention.dropout": self.backbone.encoder.blocks[1].attention.dropout,
            "backbone.encoder.blocks.1.attention.prob_dropout": self.backbone.encoder.blocks[1].attention.prob_dropout,
            "backbone.encoder.blocks.1.attention.softmax": self.backbone.encoder.blocks[1].attention.softmax,
            "backbone.encoder.blocks.1.attention.dense1": self.backbone.encoder.blocks[1].attention.dense1,
            "backbone.encoder.blocks.1.attention.dense2": self.backbone.encoder.blocks[1].attention.dense2,
            "backbone.encoder.blocks.1.attention.dense3": self.backbone.encoder.blocks[1].attention.dense3,
            "backbone.encoder.blocks.1.output": self.backbone.encoder.blocks[1].output,
            "backbone.encoder.blocks.1.output.mapping": self.backbone.encoder.blocks[1].output.mapping,
            "backbone.encoder.blocks.1.output.projection": self.backbone.encoder.blocks[1].output.projection,
            "backbone.encoder.blocks.1.output.dropout": self.backbone.encoder.blocks[1].output.dropout,
            "backbone.encoder.blocks.2": self.backbone.encoder.blocks[2],
            "backbone.encoder.blocks.2.layernorm1": self.backbone.encoder.blocks[2].layernorm1,
            "backbone.encoder.blocks.2.layernorm2": self.backbone.encoder.blocks[2].layernorm2,
            "backbone.encoder.blocks.2.attention": self.backbone.encoder.blocks[2].attention,
            "backbone.encoder.blocks.2.attention.projection": self.backbone.encoder.blocks[2].attention.projection,
            "backbone.encoder.blocks.2.attention.dropout": self.backbone.encoder.blocks[2].attention.dropout,
            "backbone.encoder.blocks.2.attention.prob_dropout": self.backbone.encoder.blocks[2].attention.prob_dropout,
            "backbone.encoder.blocks.2.attention.softmax": self.backbone.encoder.blocks[2].attention.softmax,
            "backbone.encoder.blocks.2.attention.dense1": self.backbone.encoder.blocks[2].attention.dense1,
            "backbone.encoder.blocks.2.attention.dense2": self.backbone.encoder.blocks[2].attention.dense2,
            "backbone.encoder.blocks.2.attention.dense3": self.backbone.encoder.blocks[2].attention.dense3,
            "backbone.encoder.blocks.2.output": self.backbone.encoder.blocks[2].output,
            "backbone.encoder.blocks.2.output.mapping": self.backbone.encoder.blocks[2].output.mapping,
            "backbone.encoder.blocks.2.output.projection": self.backbone.encoder.blocks[2].output.projection,
            "backbone.encoder.blocks.2.output.dropout": self.backbone.encoder.blocks[2].output.dropout,
            "backbone.encoder.blocks.3": self.backbone.encoder.blocks[3],
            "backbone.encoder.blocks.3.layernorm1": self.backbone.encoder.blocks[3].layernorm1,
            "backbone.encoder.blocks.3.layernorm2": self.backbone.encoder.blocks[3].layernorm2,
            "backbone.encoder.blocks.3.attention": self.backbone.encoder.blocks[3].attention,
            "backbone.encoder.blocks.3.attention.projection": self.backbone.encoder.blocks[3].attention.projection,
            "backbone.encoder.blocks.3.attention.dropout": self.backbone.encoder.blocks[3].attention.dropout,
            "backbone.encoder.blocks.3.attention.prob_dropout": self.backbone.encoder.blocks[3].attention.prob_dropout,
            "backbone.encoder.blocks.3.attention.softmax": self.backbone.encoder.blocks[3].attention.softmax,
            "backbone.encoder.blocks.3.attention.dense1": self.backbone.encoder.blocks[3].attention.dense1,
            "backbone.encoder.blocks.3.attention.dense2": self.backbone.encoder.blocks[3].attention.dense2,
            "backbone.encoder.blocks.3.attention.dense3": self.backbone.encoder.blocks[3].attention.dense3,
            "backbone.encoder.blocks.3.output": self.backbone.encoder.blocks[3].output,
            "backbone.encoder.blocks.3.output.mapping": self.backbone.encoder.blocks[3].output.mapping,
            "backbone.encoder.blocks.3.output.projection": self.backbone.encoder.blocks[3].output.projection,
            "backbone.encoder.blocks.3.output.dropout": self.backbone.encoder.blocks[3].output.dropout,
            "backbone.encoder.blocks.4": self.backbone.encoder.blocks[4],
            "backbone.encoder.blocks.4.layernorm1": self.backbone.encoder.blocks[4].layernorm1,
            "backbone.encoder.blocks.4.layernorm2": self.backbone.encoder.blocks[4].layernorm2,
            "backbone.encoder.blocks.4.attention": self.backbone.encoder.blocks[4].attention,
            "backbone.encoder.blocks.4.attention.projection": self.backbone.encoder.blocks[4].attention.projection,
            "backbone.encoder.blocks.4.attention.dropout": self.backbone.encoder.blocks[4].attention.dropout,
            "backbone.encoder.blocks.4.attention.prob_dropout": self.backbone.encoder.blocks[4].attention.prob_dropout,
            "backbone.encoder.blocks.4.attention.softmax": self.backbone.encoder.blocks[4].attention.softmax,
            "backbone.encoder.blocks.4.attention.dense1": self.backbone.encoder.blocks[4].attention.dense1,
            "backbone.encoder.blocks.4.attention.dense2": self.backbone.encoder.blocks[4].attention.dense2,
            "backbone.encoder.blocks.4.attention.dense3": self.backbone.encoder.blocks[4].attention.dense3,
            "backbone.encoder.blocks.4.output": self.backbone.encoder.blocks[4].output,
            "backbone.encoder.blocks.4.output.mapping": self.backbone.encoder.blocks[4].output.mapping,
            "backbone.encoder.blocks.4.output.projection": self.backbone.encoder.blocks[4].output.projection,
            "backbone.encoder.blocks.4.output.dropout": self.backbone.encoder.blocks[4].output.dropout,
            "backbone.encoder.blocks.5": self.backbone.encoder.blocks[5],
            "backbone.encoder.blocks.5.layernorm1": self.backbone.encoder.blocks[5].layernorm1,
            "backbone.encoder.blocks.5.layernorm2": self.backbone.encoder.blocks[5].layernorm2,
            "backbone.encoder.blocks.5.attention": self.backbone.encoder.blocks[5].attention,
            "backbone.encoder.blocks.5.attention.projection": self.backbone.encoder.blocks[5].attention.projection,
            "backbone.encoder.blocks.5.attention.dropout": self.backbone.encoder.blocks[5].attention.dropout,
            "backbone.encoder.blocks.5.attention.prob_dropout": self.backbone.encoder.blocks[5].attention.prob_dropout,
            "backbone.encoder.blocks.5.attention.softmax": self.backbone.encoder.blocks[5].attention.softmax,
            "backbone.encoder.blocks.5.attention.dense1": self.backbone.encoder.blocks[5].attention.dense1,
            "backbone.encoder.blocks.5.attention.dense2": self.backbone.encoder.blocks[5].attention.dense2,
            "backbone.encoder.blocks.5.attention.dense3": self.backbone.encoder.blocks[5].attention.dense3,
            "backbone.encoder.blocks.5.output": self.backbone.encoder.blocks[5].output,
            "backbone.encoder.blocks.5.output.mapping": self.backbone.encoder.blocks[5].output.mapping,
            "backbone.encoder.blocks.5.output.projection": self.backbone.encoder.blocks[5].output.projection,
            "backbone.encoder.blocks.5.output.dropout": self.backbone.encoder.blocks[5].output.dropout,
            "backbone.encoder.blocks.6": self.backbone.encoder.blocks[6],
            "backbone.encoder.blocks.6.layernorm1": self.backbone.encoder.blocks[6].layernorm1,
            "backbone.encoder.blocks.6.layernorm2": self.backbone.encoder.blocks[6].layernorm2,
            "backbone.encoder.blocks.6.attention": self.backbone.encoder.blocks[6].attention,
            "backbone.encoder.blocks.6.attention.projection": self.backbone.encoder.blocks[6].attention.projection,
            "backbone.encoder.blocks.6.attention.dropout": self.backbone.encoder.blocks[6].attention.dropout,
            "backbone.encoder.blocks.6.attention.prob_dropout": self.backbone.encoder.blocks[6].attention.prob_dropout,
            "backbone.encoder.blocks.6.attention.softmax": self.backbone.encoder.blocks[6].attention.softmax,
            "backbone.encoder.blocks.6.attention.dense1": self.backbone.encoder.blocks[6].attention.dense1,
            "backbone.encoder.blocks.6.attention.dense2": self.backbone.encoder.blocks[6].attention.dense2,
            "backbone.encoder.blocks.6.attention.dense3": self.backbone.encoder.blocks[6].attention.dense3,
            "backbone.encoder.blocks.6.output": self.backbone.encoder.blocks[6].output,
            "backbone.encoder.blocks.6.output.mapping": self.backbone.encoder.blocks[6].output.mapping,
            "backbone.encoder.blocks.6.output.projection": self.backbone.encoder.blocks[6].output.projection,
            "backbone.encoder.blocks.6.output.dropout": self.backbone.encoder.blocks[6].output.dropout,
            "backbone.encoder.blocks.7": self.backbone.encoder.blocks[7],
            "backbone.encoder.blocks.7.layernorm1": self.backbone.encoder.blocks[7].layernorm1,
            "backbone.encoder.blocks.7.layernorm2": self.backbone.encoder.blocks[7].layernorm2,
            "backbone.encoder.blocks.7.attention": self.backbone.encoder.blocks[7].attention,
            "backbone.encoder.blocks.7.attention.projection": self.backbone.encoder.blocks[7].attention.projection,
            "backbone.encoder.blocks.7.attention.dropout": self.backbone.encoder.blocks[7].attention.dropout,
            "backbone.encoder.blocks.7.attention.prob_dropout": self.backbone.encoder.blocks[7].attention.prob_dropout,
            "backbone.encoder.blocks.7.attention.softmax": self.backbone.encoder.blocks[7].attention.softmax,
            "backbone.encoder.blocks.7.attention.dense1": self.backbone.encoder.blocks[7].attention.dense1,
            "backbone.encoder.blocks.7.attention.dense2": self.backbone.encoder.blocks[7].attention.dense2,
            "backbone.encoder.blocks.7.attention.dense3": self.backbone.encoder.blocks[7].attention.dense3,
            "backbone.encoder.blocks.7.output": self.backbone.encoder.blocks[7].output,
            "backbone.encoder.blocks.7.output.mapping": self.backbone.encoder.blocks[7].output.mapping,
            "backbone.encoder.blocks.7.output.projection": self.backbone.encoder.blocks[7].output.projection,
            "backbone.encoder.blocks.7.output.dropout": self.backbone.encoder.blocks[7].output.dropout,
            # "backbone.encoder.blocks.8": self.backbone.encoder.blocks[8],
            # "backbone.encoder.blocks.8.layernorm1": self.backbone.encoder.blocks[8].layernorm1,
            # "backbone.encoder.blocks.8.layernorm2": self.backbone.encoder.blocks[8].layernorm2,
            # "backbone.encoder.blocks.8.attention": self.backbone.encoder.blocks[8].attention,
            # "backbone.encoder.blocks.8.attention.projection": self.backbone.encoder.blocks[8].attention.projection,
            # "backbone.encoder.blocks.8.attention.dropout": self.backbone.encoder.blocks[8].attention.dropout,
            # "backbone.encoder.blocks.8.attention.prob_dropout": self.backbone.encoder.blocks[8].attention.prob_dropout,
            # "backbone.encoder.blocks.8.attention.softmax": self.backbone.encoder.blocks[8].attention.softmax,
            # "backbone.encoder.blocks.8.attention.dense1": self.backbone.encoder.blocks[8].attention.dense1,
            # "backbone.encoder.blocks.8.attention.dense2": self.backbone.encoder.blocks[8].attention.dense2,
            # "backbone.encoder.blocks.8.attention.dense3": self.backbone.encoder.blocks[8].attention.dense3,
            # "backbone.encoder.blocks.8.output": self.backbone.encoder.blocks[8].output,
            # "backbone.encoder.blocks.8.output.mapping": self.backbone.encoder.blocks[8].output.mapping,
            # "backbone.encoder.blocks.8.output.projection": self.backbone.encoder.blocks[8].output.projection,
            # "backbone.encoder.blocks.8.output.dropout": self.backbone.encoder.blocks[8].output.dropout,
            # "backbone.encoder.blocks.9": self.backbone.encoder.blocks[9],
            # "backbone.encoder.blocks.9.layernorm1": self.backbone.encoder.blocks[9].layernorm1,
            # "backbone.encoder.blocks.9.layernorm2": self.backbone.encoder.blocks[9].layernorm2,
            # "backbone.encoder.blocks.9.attention": self.backbone.encoder.blocks[9].attention,
            # "backbone.encoder.blocks.9.attention.projection": self.backbone.encoder.blocks[9].attention.projection,
            # "backbone.encoder.blocks.9.attention.dropout": self.backbone.encoder.blocks[9].attention.dropout,
            # "backbone.encoder.blocks.9.attention.prob_dropout": self.backbone.encoder.blocks[9].attention.prob_dropout,
            # "backbone.encoder.blocks.9.attention.softmax": self.backbone.encoder.blocks[9].attention.softmax,
            # "backbone.encoder.blocks.9.attention.dense1": self.backbone.encoder.blocks[9].attention.dense1,
            # "backbone.encoder.blocks.9.attention.dense2": self.backbone.encoder.blocks[9].attention.dense2,
            # "backbone.encoder.blocks.9.attention.dense3": self.backbone.encoder.blocks[9].attention.dense3,
            # "backbone.encoder.blocks.9.output": self.backbone.encoder.blocks[9].output,
            # "backbone.encoder.blocks.9.output.mapping": self.backbone.encoder.blocks[9].output.mapping,
            # "backbone.encoder.blocks.9.output.projection": self.backbone.encoder.blocks[9].output.projection,
            # "backbone.encoder.blocks.9.output.dropout": self.backbone.encoder.blocks[9].output.dropout,
            # "backbone.encoder.blocks.10": self.backbone.encoder.blocks[10],
            # "backbone.encoder.blocks.10.layernorm1": self.backbone.encoder.blocks[10].layernorm1,
            # "backbone.encoder.blocks.10.layernorm2": self.backbone.encoder.blocks[10].layernorm2,
            # "backbone.encoder.blocks.10.attention": self.backbone.encoder.blocks[10].attention,
            # "backbone.encoder.blocks.10.attention.projection": self.backbone.encoder.blocks[10].attention.projection,
            # "backbone.encoder.blocks.10.attention.dropout": self.backbone.encoder.blocks[10].attention.dropout,
            # "backbone.encoder.blocks.10.attention.prob_dropout": self.backbone.encoder.blocks[
            #     10].attention.prob_dropout,
            # "backbone.encoder.blocks.10.attention.softmax": self.backbone.encoder.blocks[10].attention.softmax,
            # "backbone.encoder.blocks.10.attention.dense1": self.backbone.encoder.blocks[10].attention.dense1,
            # "backbone.encoder.blocks.10.attention.dense2": self.backbone.encoder.blocks[10].attention.dense2,
            # "backbone.encoder.blocks.10.attention.dense3": self.backbone.encoder.blocks[10].attention.dense3,
            # "backbone.encoder.blocks.10.output": self.backbone.encoder.blocks[10].output,
            # "backbone.encoder.blocks.10.output.mapping": self.backbone.encoder.blocks[10].output.mapping,
            # "backbone.encoder.blocks.10.output.projection": self.backbone.encoder.blocks[10].output.projection,
            # "backbone.encoder.blocks.10.output.dropout": self.backbone.encoder.blocks[10].output.dropout,
            # "backbone.encoder.blocks.11": self.backbone.encoder.blocks[11],
            # "backbone.encoder.blocks.11.layernorm1": self.backbone.encoder.blocks[11].layernorm1,
            # "backbone.encoder.blocks.11.layernorm2": self.backbone.encoder.blocks[11].layernorm2,
            # "backbone.encoder.blocks.11.attention": self.backbone.encoder.blocks[11].attention,
            # "backbone.encoder.blocks.11.attention.projection": self.backbone.encoder.blocks[11].attention.projection,
            # "backbone.encoder.blocks.11.attention.dropout": self.backbone.encoder.blocks[11].attention.dropout,
            # "backbone.encoder.blocks.11.attention.prob_dropout": self.backbone.encoder.blocks[
            #     11].attention.prob_dropout,
            # "backbone.encoder.blocks.11.attention.softmax": self.backbone.encoder.blocks[11].attention.softmax,
            # "backbone.encoder.blocks.11.attention.dense1": self.backbone.encoder.blocks[11].attention.dense1,
            # "backbone.encoder.blocks.11.attention.dense2": self.backbone.encoder.blocks[11].attention.dense2,
            # "backbone.encoder.blocks.11.attention.dense3": self.backbone.encoder.blocks[11].attention.dense3,
            # "backbone.encoder.blocks.11.output": self.backbone.encoder.blocks[11].output,
            # "backbone.encoder.blocks.11.output.mapping": self.backbone.encoder.blocks[11].output.mapping,
            # "backbone.encoder.blocks.11.output.projection": self.backbone.encoder.blocks[11].output.projection,
            # "backbone.encoder.blocks.11.output.dropout": self.backbone.encoder.blocks[11].output.dropout,
            # "backbone.encoder.blocks.12": self.backbone.encoder.blocks[12],
            # "backbone.encoder.blocks.12.layernorm1": self.backbone.encoder.blocks[12].layernorm1,
            # "backbone.encoder.blocks.12.layernorm2": self.backbone.encoder.blocks[12].layernorm2,
            # "backbone.encoder.blocks.12.attention": self.backbone.encoder.blocks[12].attention,
            # "backbone.encoder.blocks.12.attention.projection": self.backbone.encoder.blocks[12].attention.projection,
            # "backbone.encoder.blocks.12.attention.dropout": self.backbone.encoder.blocks[12].attention.dropout,
            # "backbone.encoder.blocks.12.attention.prob_dropout": self.backbone.encoder.blocks[
            #     12].attention.prob_dropout,
            # "backbone.encoder.blocks.12.attention.softmax": self.backbone.encoder.blocks[12].attention.softmax,
            # "backbone.encoder.blocks.12.attention.dense1": self.backbone.encoder.blocks[12].attention.dense1,
            # "backbone.encoder.blocks.12.attention.dense2": self.backbone.encoder.blocks[12].attention.dense2,
            # "backbone.encoder.blocks.12.attention.dense3": self.backbone.encoder.blocks[12].attention.dense3,
            # "backbone.encoder.blocks.12.output": self.backbone.encoder.blocks[12].output,
            # "backbone.encoder.blocks.12.output.mapping": self.backbone.encoder.blocks[12].output.mapping,
            # "backbone.encoder.blocks.12.output.projection": self.backbone.encoder.blocks[12].output.projection,
            # "backbone.encoder.blocks.12.output.dropout": self.backbone.encoder.blocks[12].output.dropout,
            # "backbone.encoder.blocks.13": self.backbone.encoder.blocks[13],
            # "backbone.encoder.blocks.13.layernorm1": self.backbone.encoder.blocks[13].layernorm1,
            # "backbone.encoder.blocks.13.layernorm2": self.backbone.encoder.blocks[13].layernorm2,
            # "backbone.encoder.blocks.13.attention": self.backbone.encoder.blocks[13].attention,
            # "backbone.encoder.blocks.13.attention.projection": self.backbone.encoder.blocks[13].attention.projection,
            # "backbone.encoder.blocks.13.attention.dropout": self.backbone.encoder.blocks[13].attention.dropout,
            # "backbone.encoder.blocks.13.attention.prob_dropout": self.backbone.encoder.blocks[
            #     13].attention.prob_dropout,
            # "backbone.encoder.blocks.13.attention.softmax": self.backbone.encoder.blocks[13].attention.softmax,
            # "backbone.encoder.blocks.13.attention.dense1": self.backbone.encoder.blocks[13].attention.dense1,
            # "backbone.encoder.blocks.13.attention.dense2": self.backbone.encoder.blocks[13].attention.dense2,
            # "backbone.encoder.blocks.13.attention.dense3": self.backbone.encoder.blocks[13].attention.dense3,
            # "backbone.encoder.blocks.13.output": self.backbone.encoder.blocks[13].output,
            # "backbone.encoder.blocks.13.output.mapping": self.backbone.encoder.blocks[13].output.mapping,
            # "backbone.encoder.blocks.13.output.projection": self.backbone.encoder.blocks[13].output.projection,
            # "backbone.encoder.blocks.13.output.dropout": self.backbone.encoder.blocks[13].output.dropout,
            # "backbone.encoder.blocks.14": self.backbone.encoder.blocks[14],
            # "backbone.encoder.blocks.14.layernorm1": self.backbone.encoder.blocks[14].layernorm1,
            # "backbone.encoder.blocks.14.layernorm2": self.backbone.encoder.blocks[14].layernorm2,
            # "backbone.encoder.blocks.14.attention": self.backbone.encoder.blocks[14].attention,
            # "backbone.encoder.blocks.14.attention.projection": self.backbone.encoder.blocks[14].attention.projection,
            # "backbone.encoder.blocks.14.attention.dropout": self.backbone.encoder.blocks[14].attention.dropout,
            # "backbone.encoder.blocks.14.attention.prob_dropout": self.backbone.encoder.blocks[
            #     14].attention.prob_dropout,
            # "backbone.encoder.blocks.14.attention.softmax": self.backbone.encoder.blocks[14].attention.softmax,
            # "backbone.encoder.blocks.14.attention.dense1": self.backbone.encoder.blocks[14].attention.dense1,
            # "backbone.encoder.blocks.14.attention.dense2": self.backbone.encoder.blocks[14].attention.dense2,
            # "backbone.encoder.blocks.14.attention.dense3": self.backbone.encoder.blocks[14].attention.dense3,
            # "backbone.encoder.blocks.14.output": self.backbone.encoder.blocks[14].output,
            # "backbone.encoder.blocks.14.output.mapping": self.backbone.encoder.blocks[14].output.mapping,
            # "backbone.encoder.blocks.14.output.projection": self.backbone.encoder.blocks[14].output.projection,
            # "backbone.encoder.blocks.14.output.dropout": self.backbone.encoder.blocks[14].output.dropout,
            # "backbone.encoder.blocks.15": self.backbone.encoder.blocks[15],
            # "backbone.encoder.blocks.15.layernorm1": self.backbone.encoder.blocks[15].layernorm1,
            # "backbone.encoder.blocks.15.layernorm2": self.backbone.encoder.blocks[15].layernorm2,
            # "backbone.encoder.blocks.15.attention": self.backbone.encoder.blocks[15].attention,
            # "backbone.encoder.blocks.15.attention.projection": self.backbone.encoder.blocks[15].attention.projection,
            # "backbone.encoder.blocks.15.attention.dropout": self.backbone.encoder.blocks[15].attention.dropout,
            # "backbone.encoder.blocks.15.attention.prob_dropout": self.backbone.encoder.blocks[
            #     15].attention.prob_dropout,
            # "backbone.encoder.blocks.15.attention.softmax": self.backbone.encoder.blocks[15].attention.softmax,
            # "backbone.encoder.blocks.15.attention.dense1": self.backbone.encoder.blocks[15].attention.dense1,
            # "backbone.encoder.blocks.15.attention.dense2": self.backbone.encoder.blocks[15].attention.dense2,
            # "backbone.encoder.blocks.15.attention.dense3": self.backbone.encoder.blocks[15].attention.dense3,
            # "backbone.encoder.blocks.15.output": self.backbone.encoder.blocks[15].output,
            # "backbone.encoder.blocks.15.output.mapping": self.backbone.encoder.blocks[15].output.mapping,
            # "backbone.encoder.blocks.15.output.projection": self.backbone.encoder.blocks[15].output.projection,
            # "backbone.encoder.blocks.15.output.dropout": self.backbone.encoder.blocks[15].output.dropout,
            # "backbone.encoder.blocks.16": self.backbone.encoder.blocks[16],
            # "backbone.encoder.blocks.16.layernorm1": self.backbone.encoder.blocks[16].layernorm1,
            # "backbone.encoder.blocks.16.layernorm2": self.backbone.encoder.blocks[16].layernorm2,
            # "backbone.encoder.blocks.16.attention": self.backbone.encoder.blocks[16].attention,
            # "backbone.encoder.blocks.16.attention.projection": self.backbone.encoder.blocks[16].attention.projection,
            # "backbone.encoder.blocks.16.attention.dropout": self.backbone.encoder.blocks[16].attention.dropout,
            # "backbone.encoder.blocks.16.attention.prob_dropout": self.backbone.encoder.blocks[
            #     16].attention.prob_dropout,
            # "backbone.encoder.blocks.16.attention.softmax": self.backbone.encoder.blocks[16].attention.softmax,
            # "backbone.encoder.blocks.16.attention.dense1": self.backbone.encoder.blocks[16].attention.dense1,
            # "backbone.encoder.blocks.16.attention.dense2": self.backbone.encoder.blocks[16].attention.dense2,
            # "backbone.encoder.blocks.16.attention.dense3": self.backbone.encoder.blocks[16].attention.dense3,
            # "backbone.encoder.blocks.16.output": self.backbone.encoder.blocks[16].output,
            # "backbone.encoder.blocks.16.output.mapping": self.backbone.encoder.blocks[16].output.mapping,
            # "backbone.encoder.blocks.16.output.projection": self.backbone.encoder.blocks[16].output.projection,
            # "backbone.encoder.blocks.16.output.dropout": self.backbone.encoder.blocks[16].output.dropout,
            # "backbone.encoder.blocks.17": self.backbone.encoder.blocks[17],
            # "backbone.encoder.blocks.17.layernorm1": self.backbone.encoder.blocks[17].layernorm1,
            # "backbone.encoder.blocks.17.layernorm2": self.backbone.encoder.blocks[17].layernorm2,
            # "backbone.encoder.blocks.17.attention": self.backbone.encoder.blocks[17].attention,
            # "backbone.encoder.blocks.17.attention.projection": self.backbone.encoder.blocks[17].attention.projection,
            # "backbone.encoder.blocks.17.attention.dropout": self.backbone.encoder.blocks[17].attention.dropout,
            # "backbone.encoder.blocks.17.attention.prob_dropout": self.backbone.encoder.blocks[
            #     17].attention.prob_dropout,
            # "backbone.encoder.blocks.17.attention.softmax": self.backbone.encoder.blocks[17].attention.softmax,
            # "backbone.encoder.blocks.17.attention.dense1": self.backbone.encoder.blocks[17].attention.dense1,
            # "backbone.encoder.blocks.17.attention.dense2": self.backbone.encoder.blocks[17].attention.dense2,
            # "backbone.encoder.blocks.17.attention.dense3": self.backbone.encoder.blocks[17].attention.dense3,
            # "backbone.encoder.blocks.17.output": self.backbone.encoder.blocks[17].output,
            # "backbone.encoder.blocks.17.output.mapping": self.backbone.encoder.blocks[17].output.mapping,
            # "backbone.encoder.blocks.17.output.projection": self.backbone.encoder.blocks[17].output.projection,
            # "backbone.encoder.blocks.17.output.dropout": self.backbone.encoder.blocks[17].output.dropout,
            # "backbone.encoder.blocks.18": self.backbone.encoder.blocks[18],
            # "backbone.encoder.blocks.18.layernorm1": self.backbone.encoder.blocks[18].layernorm1,
            # "backbone.encoder.blocks.18.layernorm2": self.backbone.encoder.blocks[18].layernorm2,
            # "backbone.encoder.blocks.18.attention": self.backbone.encoder.blocks[18].attention,
            # "backbone.encoder.blocks.18.attention.projection": self.backbone.encoder.blocks[18].attention.projection,
            # "backbone.encoder.blocks.18.attention.dropout": self.backbone.encoder.blocks[18].attention.dropout,
            # "backbone.encoder.blocks.18.attention.prob_dropout": self.backbone.encoder.blocks[
            #     18].attention.prob_dropout,
            # "backbone.encoder.blocks.18.attention.softmax": self.backbone.encoder.blocks[18].attention.softmax,
            # "backbone.encoder.blocks.18.attention.dense1": self.backbone.encoder.blocks[18].attention.dense1,
            # "backbone.encoder.blocks.18.attention.dense2": self.backbone.encoder.blocks[18].attention.dense2,
            # "backbone.encoder.blocks.18.attention.dense3": self.backbone.encoder.blocks[18].attention.dense3,
            # "backbone.encoder.blocks.18.output": self.backbone.encoder.blocks[18].output,
            # "backbone.encoder.blocks.18.output.mapping": self.backbone.encoder.blocks[18].output.mapping,
            # "backbone.encoder.blocks.18.output.projection": self.backbone.encoder.blocks[18].output.projection,
            # "backbone.encoder.blocks.18.output.dropout": self.backbone.encoder.blocks[18].output.dropout,
            # "backbone.encoder.blocks.19": self.backbone.encoder.blocks[19],
            # "backbone.encoder.blocks.19.layernorm1": self.backbone.encoder.blocks[19].layernorm1,
            # "backbone.encoder.blocks.19.layernorm2": self.backbone.encoder.blocks[19].layernorm2,
            # "backbone.encoder.blocks.19.attention": self.backbone.encoder.blocks[19].attention,
            # "backbone.encoder.blocks.19.attention.projection": self.backbone.encoder.blocks[19].attention.projection,
            # "backbone.encoder.blocks.19.attention.dropout": self.backbone.encoder.blocks[19].attention.dropout,
            # "backbone.encoder.blocks.19.attention.prob_dropout": self.backbone.encoder.blocks[
            #     19].attention.prob_dropout,
            # "backbone.encoder.blocks.19.attention.softmax": self.backbone.encoder.blocks[19].attention.softmax,
            # "backbone.encoder.blocks.19.attention.dense1": self.backbone.encoder.blocks[19].attention.dense1,
            # "backbone.encoder.blocks.19.attention.dense2": self.backbone.encoder.blocks[19].attention.dense2,
            # "backbone.encoder.blocks.19.attention.dense3": self.backbone.encoder.blocks[19].attention.dense3,
            # "backbone.encoder.blocks.19.output": self.backbone.encoder.blocks[19].output,
            # "backbone.encoder.blocks.19.output.mapping": self.backbone.encoder.blocks[19].output.mapping,
            # "backbone.encoder.blocks.19.output.projection": self.backbone.encoder.blocks[19].output.projection,
            # "backbone.encoder.blocks.19.output.dropout": self.backbone.encoder.blocks[19].output.dropout,
            # "backbone.encoder.blocks.20": self.backbone.encoder.blocks[20],
            # "backbone.encoder.blocks.20.layernorm1": self.backbone.encoder.blocks[20].layernorm1,
            # "backbone.encoder.blocks.20.layernorm2": self.backbone.encoder.blocks[20].layernorm2,
            # "backbone.encoder.blocks.20.attention": self.backbone.encoder.blocks[20].attention,
            # "backbone.encoder.blocks.20.attention.projection": self.backbone.encoder.blocks[20].attention.projection,
            # "backbone.encoder.blocks.20.attention.dropout": self.backbone.encoder.blocks[20].attention.dropout,
            # "backbone.encoder.blocks.20.attention.prob_dropout": self.backbone.encoder.blocks[
            #     20].attention.prob_dropout,
            # "backbone.encoder.blocks.20.attention.softmax": self.backbone.encoder.blocks[20].attention.softmax,
            # "backbone.encoder.blocks.20.attention.dense1": self.backbone.encoder.blocks[20].attention.dense1,
            # "backbone.encoder.blocks.20.attention.dense2": self.backbone.encoder.blocks[20].attention.dense2,
            # "backbone.encoder.blocks.20.attention.dense3": self.backbone.encoder.blocks[20].attention.dense3,
            # "backbone.encoder.blocks.20.output": self.backbone.encoder.blocks[20].output,
            # "backbone.encoder.blocks.20.output.mapping": self.backbone.encoder.blocks[20].output.mapping,
            # "backbone.encoder.blocks.20.output.projection": self.backbone.encoder.blocks[20].output.projection,
            # "backbone.encoder.blocks.20.output.dropout": self.backbone.encoder.blocks[20].output.dropout,
            # "backbone.encoder.blocks.21": self.backbone.encoder.blocks[21],
            # "backbone.encoder.blocks.21.layernorm1": self.backbone.encoder.blocks[21].layernorm1,
            # "backbone.encoder.blocks.21.layernorm2": self.backbone.encoder.blocks[21].layernorm2,
            # "backbone.encoder.blocks.21.attention": self.backbone.encoder.blocks[21].attention,
            # "backbone.encoder.blocks.21.attention.projection": self.backbone.encoder.blocks[21].attention.projection,
            # "backbone.encoder.blocks.21.attention.dropout": self.backbone.encoder.blocks[21].attention.dropout,
            # "backbone.encoder.blocks.21.attention.prob_dropout": self.backbone.encoder.blocks[
            #     21].attention.prob_dropout,
            # "backbone.encoder.blocks.21.attention.softmax": self.backbone.encoder.blocks[21].attention.softmax,
            # "backbone.encoder.blocks.21.attention.dense1": self.backbone.encoder.blocks[21].attention.dense1,
            # "backbone.encoder.blocks.21.attention.dense2": self.backbone.encoder.blocks[21].attention.dense2,
            # "backbone.encoder.blocks.21.attention.dense3": self.backbone.encoder.blocks[21].attention.dense3,
            # "backbone.encoder.blocks.21.output": self.backbone.encoder.blocks[21].output,
            # "backbone.encoder.blocks.21.output.mapping": self.backbone.encoder.blocks[21].output.mapping,
            # "backbone.encoder.blocks.21.output.projection": self.backbone.encoder.blocks[21].output.projection,
            # "backbone.encoder.blocks.21.output.dropout": self.backbone.encoder.blocks[21].output.dropout,
            # "backbone.encoder.blocks.22": self.backbone.encoder.blocks[22],
            # "backbone.encoder.blocks.22.layernorm1": self.backbone.encoder.blocks[22].layernorm1,
            # "backbone.encoder.blocks.22.layernorm2": self.backbone.encoder.blocks[22].layernorm2,
            # "backbone.encoder.blocks.22.attention": self.backbone.encoder.blocks[22].attention,
            # "backbone.encoder.blocks.22.attention.projection": self.backbone.encoder.blocks[22].attention.projection,
            # "backbone.encoder.blocks.22.attention.dropout": self.backbone.encoder.blocks[22].attention.dropout,
            # "backbone.encoder.blocks.22.attention.prob_dropout": self.backbone.encoder.blocks[
            #     22].attention.prob_dropout,
            # "backbone.encoder.blocks.22.attention.softmax": self.backbone.encoder.blocks[22].attention.softmax,
            # "backbone.encoder.blocks.22.attention.dense1": self.backbone.encoder.blocks[22].attention.dense1,
            # "backbone.encoder.blocks.22.attention.dense2": self.backbone.encoder.blocks[22].attention.dense2,
            # "backbone.encoder.blocks.22.attention.dense3": self.backbone.encoder.blocks[22].attention.dense3,
            # "backbone.encoder.blocks.22.output": self.backbone.encoder.blocks[22].output,
            # "backbone.encoder.blocks.22.output.mapping": self.backbone.encoder.blocks[22].output.mapping,
            # "backbone.encoder.blocks.22.output.projection": self.backbone.encoder.blocks[22].output.projection,
            # "backbone.encoder.blocks.22.output.dropout": self.backbone.encoder.blocks[22].output.dropout,
            # "backbone.encoder.blocks.23": self.backbone.encoder.blocks[23],
            # "backbone.encoder.blocks.23.layernorm1": self.backbone.encoder.blocks[23].layernorm1,
            # "backbone.encoder.blocks.23.layernorm2": self.backbone.encoder.blocks[23].layernorm2,
            # "backbone.encoder.blocks.23.attention": self.backbone.encoder.blocks[23].attention,
            # "backbone.encoder.blocks.23.attention.projection": self.backbone.encoder.blocks[23].attention.projection,
            # "backbone.encoder.blocks.23.attention.dropout": self.backbone.encoder.blocks[23].attention.dropout,
            # "backbone.encoder.blocks.23.attention.prob_dropout": self.backbone.encoder.blocks[
            #     23].attention.prob_dropout,
            # "backbone.encoder.blocks.23.attention.softmax": self.backbone.encoder.blocks[23].attention.softmax,
            # "backbone.encoder.blocks.23.attention.dense1": self.backbone.encoder.blocks[23].attention.dense1,
            # "backbone.encoder.blocks.23.attention.dense2": self.backbone.encoder.blocks[23].attention.dense2,
            # "backbone.encoder.blocks.23.attention.dense3": self.backbone.encoder.blocks[23].attention.dense3,
            # "backbone.encoder.blocks.23.output": self.backbone.encoder.blocks[23].output,
            # "backbone.encoder.blocks.23.output.mapping": self.backbone.encoder.blocks[23].output.mapping,
            # "backbone.encoder.blocks.23.output.projection": self.backbone.encoder.blocks[23].output.projection,
            # "backbone.encoder.blocks.23.output.dropout": self.backbone.encoder.blocks[23].output.dropout,
            "backbone.layernorm": self.backbone.layernorm,
            "head": self.head,
        }



    def create_positional_encodings(self, input_ids):
        # Extract the batch size and sequence length from the shape of input_ids
        batch_size, seq_length = input_ids.size()

        # Create a range tensor from 0 to seq_length-1
        input_position = torch.arange(seq_length, device=input_ids.device)

        # Tile this range tensor across the batch dimension
        input_position = input_position.unsqueeze(0).repeat(batch_size, 1)

        return input_position

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids):
        # position_ids = self.create_positional_encodings(input_ids)
        # position_ids = torch.clamp(position_ids, max=511)
        # embeddings1 = self.embeddings(input_ids)
        # print("type embeddings1", type(embeddings1))
        # embeddings2 = self.position_embeddings(position_ids)
        # print("type embeddings2", type(embeddings2))
        # embeddings = embeddings1 + embeddings2
        # hidden_states = embeddings
        # for layer in self.transformer_layers:
        #     hidden_states = layer(hidden_states)
        # logits = self.linear(hidden_states)
        # return logits
        tokens = input_ids[:, :-1]
        input_mask = torch.ne(tokens, self.eos_token).to(torch.float32)
        output_states, _, embedding_table = self.backbone(tokens, input_mask)
        logits = self.head(output_states, embedding_table)
        return logits
        # return output_states

    def get_order(self, layer_name):
        if layer_name not in self.orders.keys():
            return False
        return self.orders[layer_name]

    def set_order(self, layer_name, order):
        if layer_name not in self.orders.keys():
            return False
        self.orders[layer_name] = order

    def get_outshape(self, layer_name):

        if layer_name not in self.out_shapes.keys():
            return False

        return self.out_shapes[layer_name]

    def set_outshape(self, layer_name, out):

        if layer_name not in self.out_shapes.keys():
            return False

        self.out_shapes[layer_name] = out

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name, out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name] = out

    def set_Basic_OPS(self, b):
        self.Basic_OPS = b

    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self, c):
        self.Cascade_OPs = c

    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]

    def set_layers(self,layer_name,new_layer):
        if 'backbone' == layer_name:
            self.backbone= new_layer
            self.layer_names["backbone"]=new_layer

        elif 'backbone.get_attention_mask' == layer_name:
            self.backbone.get_attention_mask= new_layer
            self.layer_names["backbone.get_attention_mask"]=new_layer

        elif 'backbone.word_embedding' == layer_name:
            self.backbone.word_embedding= new_layer
            self.layer_names["backbone.word_embedding"]=new_layer

        elif 'backbone.position_embedding' == layer_name:
            self.backbone.position_embedding= new_layer
            self.layer_names["backbone.position_embedding"]=new_layer

        elif 'backbone.encoder' == layer_name:
            self.backbone.encoder= new_layer
            self.layer_names["backbone.encoder"]=new_layer

        elif 'backbone.encoder.blocks' == layer_name:
            self.backbone.encoder.blocks= new_layer
            self.layer_names["backbone.encoder.blocks"]=new_layer

        elif 'backbone.encoder.blocks.0' == layer_name:
            self.backbone.encoder.blocks[0]= new_layer
            self.layer_names["backbone.encoder.blocks.0"]=new_layer

        elif 'backbone.encoder.blocks.0.layernorm1' == layer_name:
            self.backbone.encoder.blocks[0].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.0.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.0.layernorm2' == layer_name:
            self.backbone.encoder.blocks[0].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.0.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.0.attention' == layer_name:
            self.backbone.encoder.blocks[0].attention= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention"]=new_layer

        elif 'backbone.encoder.blocks.0.attention.projection' == layer_name:
            self.backbone.encoder.blocks[0].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.0.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[0].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.0.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[0].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.0.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[0].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.0.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[0].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.0.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[0].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.0.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[0].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.0.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.0.output' == layer_name:
            self.backbone.encoder.blocks[0].output= new_layer
            self.layer_names["backbone.encoder.blocks.0.output"]=new_layer

        elif 'backbone.encoder.blocks.0.output.hidden_act' == layer_name:
            self.backbone.encoder.blocks[0].output.hidden_act= new_layer
            self.layer_names["backbone.encoder.blocks.0.output.hidden_act"]=new_layer

        elif 'backbone.encoder.blocks.0.output.mapping' == layer_name:
            self.backbone.encoder.blocks[0].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.0.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.0.output.projection' == layer_name:
            self.backbone.encoder.blocks[0].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.0.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.0.output.dropout' == layer_name:
            self.backbone.encoder.blocks[0].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.0.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.1' == layer_name:
            self.backbone.encoder.blocks[1]= new_layer
            self.layer_names["backbone.encoder.blocks.1"]=new_layer

        elif 'backbone.encoder.blocks.1.layernorm1' == layer_name:
            self.backbone.encoder.blocks[1].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.1.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.1.layernorm2' == layer_name:
            self.backbone.encoder.blocks[1].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.1.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.1.attention' == layer_name:
            self.backbone.encoder.blocks[1].attention= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention"]=new_layer

        elif 'backbone.encoder.blocks.1.attention.projection' == layer_name:
            self.backbone.encoder.blocks[1].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.1.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[1].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.1.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[1].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.1.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[1].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention.softmax"]=new_layer


        elif 'backbone.encoder.blocks.1.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[1].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.1.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[1].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.1.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[1].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.1.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.1.output' == layer_name:
            self.backbone.encoder.blocks[1].output= new_layer
            self.layer_names["backbone.encoder.blocks.1.output"]=new_layer

        elif 'backbone.encoder.blocks.1.output.mapping' == layer_name:
            self.backbone.encoder.blocks[1].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.1.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.1.output.projection' == layer_name:
            self.backbone.encoder.blocks[1].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.1.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.1.output.dropout' == layer_name:
            self.backbone.encoder.blocks[1].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.1.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.2' == layer_name:
            self.backbone.encoder.blocks[2]= new_layer
            self.layer_names["backbone.encoder.blocks.2"]=new_layer

        elif 'backbone.encoder.blocks.2.layernorm1' == layer_name:
            self.backbone.encoder.blocks[2].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.2.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.2.layernorm2' == layer_name:
            self.backbone.encoder.blocks[2].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.2.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.2.attention' == layer_name:
            self.backbone.encoder.blocks[2].attention= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention"]=new_layer

        elif 'backbone.encoder.blocks.2.attention.projection' == layer_name:
            self.backbone.encoder.blocks[2].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.2.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[2].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.2.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[2].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.2.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[2].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.2.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[2].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.2.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[2].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.2.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[2].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.2.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.2.output' == layer_name:
            self.backbone.encoder.blocks[2].output= new_layer
            self.layer_names["backbone.encoder.blocks.2.output"]=new_layer

        elif 'backbone.encoder.blocks.2.output.mapping' == layer_name:
            self.backbone.encoder.blocks[2].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.2.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.2.output.projection' == layer_name:
            self.backbone.encoder.blocks[2].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.2.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.2.output.dropout' == layer_name:
            self.backbone.encoder.blocks[2].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.2.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.3' == layer_name:
            self.backbone.encoder.blocks[3]= new_layer
            self.layer_names["backbone.encoder.blocks.3"]=new_layer

        elif 'backbone.encoder.blocks.3.layernorm1' == layer_name:
            self.backbone.encoder.blocks[3].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.3.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.3.layernorm2' == layer_name:
            self.backbone.encoder.blocks[3].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.3.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.3.attention' == layer_name:
            self.backbone.encoder.blocks[3].attention= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention"]=new_layer

        elif 'backbone.encoder.blocks.3.attention.projection' == layer_name:
            self.backbone.encoder.blocks[3].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.3.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[3].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.3.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[3].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.3.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[3].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.3.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[3].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.3.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[3].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.3.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[3].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.3.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.3.output' == layer_name:
            self.backbone.encoder.blocks[3].output= new_layer
            self.layer_names["backbone.encoder.blocks.3.output"]=new_layer

        elif 'backbone.encoder.blocks.3.output.mapping' == layer_name:
            self.backbone.encoder.blocks[3].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.3.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.3.output.projection' == layer_name:
            self.backbone.encoder.blocks[3].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.3.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.3.output.dropout' == layer_name:
            self.backbone.encoder.blocks[3].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.3.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.4' == layer_name:
            self.backbone.encoder.blocks[4]= new_layer
            self.layer_names["backbone.encoder.blocks.4"]=new_layer

        elif 'backbone.encoder.blocks.4.layernorm1' == layer_name:
            self.backbone.encoder.blocks[4].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.4.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.4.layernorm2' == layer_name:
            self.backbone.encoder.blocks[4].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.4.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.4.attention' == layer_name:
            self.backbone.encoder.blocks[4].attention= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention"]=new_layer

        elif 'backbone.encoder.blocks.4.attention.projection' == layer_name:
            self.backbone.encoder.blocks[4].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.4.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[4].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.4.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[4].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.4.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[4].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.4.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[4].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.4.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[4].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.4.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[4].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.4.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.4.output' == layer_name:
            self.backbone.encoder.blocks[4].output= new_layer
            self.layer_names["backbone.encoder.blocks.4.output"]=new_layer

        elif 'backbone.encoder.blocks.4.output.mapping' == layer_name:
            self.backbone.encoder.blocks[4].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.4.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.4.output.projection' == layer_name:
            self.backbone.encoder.blocks[4].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.4.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.4.output.dropout' == layer_name:
            self.backbone.encoder.blocks[4].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.4.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.5' == layer_name:
            self.backbone.encoder.blocks[5]= new_layer
            self.layer_names["backbone.encoder.blocks.5"]=new_layer

        elif 'backbone.encoder.blocks.5.layernorm1' == layer_name:
            self.backbone.encoder.blocks[5].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.5.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.5.layernorm2' == layer_name:
            self.backbone.encoder.blocks[5].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.5.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.5.attention' == layer_name:
            self.backbone.encoder.blocks[5].attention= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention"]=new_layer

        elif 'backbone.encoder.blocks.5.attention.projection' == layer_name:
            self.backbone.encoder.blocks[5].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.5.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[5].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.5.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[5].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.5.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[5].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.5.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[5].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.5.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[5].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.5.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[5].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.5.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.5.output' == layer_name:
            self.backbone.encoder.blocks[5].output= new_layer
            self.layer_names["backbone.encoder.blocks.5.output"]=new_layer

        elif 'backbone.encoder.blocks.5.output.mapping' == layer_name:
            self.backbone.encoder.blocks[5].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.5.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.5.output.projection' == layer_name:
            self.backbone.encoder.blocks[5].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.5.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.5.output.dropout' == layer_name:
            self.backbone.encoder.blocks[5].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.5.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.6' == layer_name:
            self.backbone.encoder.blocks[6]= new_layer
            self.layer_names["backbone.encoder.blocks.6"]=new_layer

        elif 'backbone.encoder.blocks.6.layernorm1' == layer_name:
            self.backbone.encoder.blocks[6].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.6.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.6.layernorm2' == layer_name:
            self.backbone.encoder.blocks[6].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.6.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.6.attention' == layer_name:
            self.backbone.encoder.blocks[6].attention= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention"]=new_layer

        elif 'backbone.encoder.blocks.6.attention.projection' == layer_name:
            self.backbone.encoder.blocks[6].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.6.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[6].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.6.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[6].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.6.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[6].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.6.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[6].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.6.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[6].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.6.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[6].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.6.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.6.output' == layer_name:
            self.backbone.encoder.blocks[6].output= new_layer
            self.layer_names["backbone.encoder.blocks.6.output"]=new_layer

        elif 'backbone.encoder.blocks.6.output.mapping' == layer_name:
            self.backbone.encoder.blocks[6].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.6.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.6.output.projection' == layer_name:
            self.backbone.encoder.blocks[6].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.6.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.6.output.dropout' == layer_name:
            self.backbone.encoder.blocks[6].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.6.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.7' == layer_name:
            self.backbone.encoder.blocks[7]= new_layer
            self.layer_names["backbone.encoder.blocks.7"]=new_layer

        elif 'backbone.encoder.blocks.7.layernorm1' == layer_name:
            self.backbone.encoder.blocks[7].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.7.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.7.layernorm2' == layer_name:
            self.backbone.encoder.blocks[7].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.7.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.7.attention' == layer_name:
            self.backbone.encoder.blocks[7].attention= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention"]=new_layer

        elif 'backbone.encoder.blocks.7.attention.projection' == layer_name:
            self.backbone.encoder.blocks[7].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.7.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[7].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.7.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[7].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.7.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[7].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.7.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[7].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.7.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[7].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.7.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[7].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.7.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.7.output' == layer_name:
            self.backbone.encoder.blocks[7].output= new_layer
            self.layer_names["backbone.encoder.blocks.7.output"]=new_layer

        elif 'backbone.encoder.blocks.7.output.mapping' == layer_name:
            self.backbone.encoder.blocks[7].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.7.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.7.output.projection' == layer_name:
            self.backbone.encoder.blocks[7].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.7.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.7.output.dropout' == layer_name:
            self.backbone.encoder.blocks[7].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.7.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.8' == layer_name:
            self.backbone.encoder.blocks[8]= new_layer
            self.layer_names["backbone.encoder.blocks.8"]=new_layer

        elif 'backbone.encoder.blocks.8.layernorm1' == layer_name:
            self.backbone.encoder.blocks[8].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.8.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.8.layernorm2' == layer_name:
            self.backbone.encoder.blocks[8].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.8.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.8.attention' == layer_name:
            self.backbone.encoder.blocks[8].attention= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.projection' == layer_name:
            self.backbone.encoder.blocks[8].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[8].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[8].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[8].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[8].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[8].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[8].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.8.output' == layer_name:
            self.backbone.encoder.blocks[8].output= new_layer
            self.layer_names["backbone.encoder.blocks.8.output"]=new_layer

        elif 'backbone.encoder.blocks.8.output.mapping' == layer_name:
            self.backbone.encoder.blocks[8].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.8.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.8.output.projection' == layer_name:
            self.backbone.encoder.blocks[8].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.8.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.8.output.dropout' == layer_name:
            self.backbone.encoder.blocks[8].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.8.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.9' == layer_name:
            self.backbone.encoder.blocks[9]= new_layer
            self.layer_names["backbone.encoder.blocks.9"]=new_layer

        elif 'backbone.encoder.blocks.9.layernorm1' == layer_name:
            self.backbone.encoder.blocks[9].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.9.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.9.layernorm2' == layer_name:
            self.backbone.encoder.blocks[9].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.9.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.9.attention' == layer_name:
            self.backbone.encoder.blocks[9].attention= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention"]=new_layer

        elif 'backbone.encoder.blocks.9.attention.projection' == layer_name:
            self.backbone.encoder.blocks[9].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.9.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[9].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.9.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[9].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.9.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[9].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.9.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[9].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.9.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[9].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.9.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[9].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.9.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.9.output' == layer_name:
            self.backbone.encoder.blocks[9].output= new_layer
            self.layer_names["backbone.encoder.blocks.9.output"]=new_layer

        elif 'backbone.encoder.blocks.9.output.mapping' == layer_name:
            self.backbone.encoder.blocks[9].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.9.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.9.output.projection' == layer_name:
            self.backbone.encoder.blocks[9].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.9.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.9.output.dropout' == layer_name:
            self.backbone.encoder.blocks[9].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.9.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.10' == layer_name:
            self.backbone.encoder.blocks[10]= new_layer
            self.layer_names["backbone.encoder.blocks.10"]=new_layer

        elif 'backbone.encoder.blocks.10.layernorm1' == layer_name:
            self.backbone.encoder.blocks[10].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.10.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.10.layernorm2' == layer_name:
            self.backbone.encoder.blocks[10].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.10.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.10.attention' == layer_name:
            self.backbone.encoder.blocks[10].attention= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention"]=new_layer

        elif 'backbone.encoder.blocks.10.attention.projection' == layer_name:
            self.backbone.encoder.blocks[10].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.10.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[10].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.10.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[10].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.10.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[10].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.10.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[10].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.10.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[10].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.10.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[10].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.10.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.10.output' == layer_name:
            self.backbone.encoder.blocks[10].output= new_layer
            self.layer_names["backbone.encoder.blocks.10.output"]=new_layer

        elif 'backbone.encoder.blocks.10.output.mapping' == layer_name:
            self.backbone.encoder.blocks[10].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.10.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.10.output.projection' == layer_name:
            self.backbone.encoder.blocks[10].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.10.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.10.output.dropout' == layer_name:
            self.backbone.encoder.blocks[10].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.10.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.11' == layer_name:
            self.backbone.encoder.blocks[11]= new_layer
            self.layer_names["backbone.encoder.blocks.11"]=new_layer

        elif 'backbone.encoder.blocks.11.layernorm1' == layer_name:
            self.backbone.encoder.blocks[11].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.11.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.11.layernorm2' == layer_name:
            self.backbone.encoder.blocks[11].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.11.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.11.attention' == layer_name:
            self.backbone.encoder.blocks[11].attention= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention"]=new_layer

        elif 'backbone.encoder.blocks.11.attention.projection' == layer_name:
            self.backbone.encoder.blocks[11].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.11.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[11].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.11.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[11].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.11.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[11].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.11.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[11].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.11.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[11].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.11.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[11].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.11.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.11.output' == layer_name:
            self.backbone.encoder.blocks[11].output= new_layer
            self.layer_names["backbone.encoder.blocks.11.output"]=new_layer

        elif 'backbone.encoder.blocks.11.output.mapping' == layer_name:
            self.backbone.encoder.blocks[11].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.11.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.11.output.projection' == layer_name:
            self.backbone.encoder.blocks[11].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.11.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.11.output.dropout' == layer_name:
            self.backbone.encoder.blocks[11].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.11.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.12' == layer_name:
            self.backbone.encoder.blocks[12]= new_layer
            self.layer_names["backbone.encoder.blocks.12"]=new_layer

        elif 'backbone.encoder.blocks.12.layernorm1' == layer_name:
            self.backbone.encoder.blocks[12].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.12.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.12.layernorm2' == layer_name:
            self.backbone.encoder.blocks[12].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.12.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.12.attention' == layer_name:
            self.backbone.encoder.blocks[12].attention= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention"]=new_layer

        elif 'backbone.encoder.blocks.12.attention.projection' == layer_name:
            self.backbone.encoder.blocks[12].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.12.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[12].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.12.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[12].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.12.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[12].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.12.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[12].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.12.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[12].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.12.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[12].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.12.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.12.output' == layer_name:
            self.backbone.encoder.blocks[12].output= new_layer
            self.layer_names["backbone.encoder.blocks.12.output"]=new_layer

        elif 'backbone.encoder.blocks.12.output.mapping' == layer_name:
            self.backbone.encoder.blocks[12].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.12.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.12.output.projection' == layer_name:
            self.backbone.encoder.blocks[12].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.12.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.12.output.dropout' == layer_name:
            self.backbone.encoder.blocks[12].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.12.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.13' == layer_name:
            self.backbone.encoder.blocks[13]= new_layer
            self.layer_names["backbone.encoder.blocks.13"]=new_layer

        elif 'backbone.encoder.blocks.13.layernorm1' == layer_name:
            self.backbone.encoder.blocks[13].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.13.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.13.layernorm2' == layer_name:
            self.backbone.encoder.blocks[13].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.13.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.13.attention' == layer_name:
            self.backbone.encoder.blocks[13].attention= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention"]=new_layer

        elif 'backbone.encoder.blocks.13.attention.projection' == layer_name:
            self.backbone.encoder.blocks[13].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.13.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[13].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.13.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[13].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.13.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[13].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.13.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[13].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.13.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[13].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.13.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[13].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.13.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.13.output' == layer_name:
            self.backbone.encoder.blocks[13].output= new_layer
            self.layer_names["backbone.encoder.blocks.13.output"]=new_layer

        elif 'backbone.encoder.blocks.13.output.mapping' == layer_name:
            self.backbone.encoder.blocks[13].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.13.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.13.output.projection' == layer_name:
            self.backbone.encoder.blocks[13].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.13.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.13.output.dropout' == layer_name:
            self.backbone.encoder.blocks[13].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.13.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.14' == layer_name:
            self.backbone.encoder.blocks[14]= new_layer
            self.layer_names["backbone.encoder.blocks.14"]=new_layer

        elif 'backbone.encoder.blocks.14.layernorm1' == layer_name:
            self.backbone.encoder.blocks[14].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.14.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.14.layernorm2' == layer_name:
            self.backbone.encoder.blocks[14].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.14.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.14.attention' == layer_name:
            self.backbone.encoder.blocks[14].attention= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention"]=new_layer

        elif 'backbone.encoder.blocks.14.attention.projection' == layer_name:
            self.backbone.encoder.blocks[14].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.14.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[14].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.14.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[14].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.14.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[14].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.14.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[14].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.14.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[14].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.14.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[14].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.14.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.14.output' == layer_name:
            self.backbone.encoder.blocks[14].output= new_layer
            self.layer_names["backbone.encoder.blocks.14.output"]=new_layer

        elif 'backbone.encoder.blocks.14.output.mapping' == layer_name:
            self.backbone.encoder.blocks[14].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.14.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.14.output.projection' == layer_name:
            self.backbone.encoder.blocks[14].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.14.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.14.output.dropout' == layer_name:
            self.backbone.encoder.blocks[14].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.14.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.15' == layer_name:
            self.backbone.encoder.blocks[15]= new_layer
            self.layer_names["backbone.encoder.blocks.15"]=new_layer

        elif 'backbone.encoder.blocks.15.layernorm1' == layer_name:
            self.backbone.encoder.blocks[15].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.15.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.15.layernorm2' == layer_name:
            self.backbone.encoder.blocks[15].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.15.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.15.attention' == layer_name:
            self.backbone.encoder.blocks[15].attention= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention"]=new_layer

        elif 'backbone.encoder.blocks.15.attention.projection' == layer_name:
            self.backbone.encoder.blocks[15].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.15.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[15].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.15.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[15].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.15.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[15].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.15.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[15].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.15.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[15].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.15.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[15].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.15.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.15.output' == layer_name:
            self.backbone.encoder.blocks[15].output= new_layer
            self.layer_names["backbone.encoder.blocks.15.output"]=new_layer

        elif 'backbone.encoder.blocks.15.output.mapping' == layer_name:
            self.backbone.encoder.blocks[15].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.15.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.15.output.projection' == layer_name:
            self.backbone.encoder.blocks[15].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.15.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.15.output.dropout' == layer_name:
            self.backbone.encoder.blocks[15].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.15.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.16' == layer_name:
            self.backbone.encoder.blocks[16]= new_layer
            self.layer_names["backbone.encoder.blocks.16"]=new_layer

        elif 'backbone.encoder.blocks.16.layernorm1' == layer_name:
            self.backbone.encoder.blocks[16].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.16.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.16.layernorm2' == layer_name:
            self.backbone.encoder.blocks[16].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.16.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.16.attention' == layer_name:
            self.backbone.encoder.blocks[16].attention= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention"]=new_layer

        elif 'backbone.encoder.blocks.16.attention.projection' == layer_name:
            self.backbone.encoder.blocks[16].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.16.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[16].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.16.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[16].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.16.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[16].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.16.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[16].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.16.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[16].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.16.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[16].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.16.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.16.output' == layer_name:
            self.backbone.encoder.blocks[16].output= new_layer
            self.layer_names["backbone.encoder.blocks.16.output"]=new_layer

        elif 'backbone.encoder.blocks.16.output.mapping' == layer_name:
            self.backbone.encoder.blocks[16].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.16.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.16.output.projection' == layer_name:
            self.backbone.encoder.blocks[16].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.16.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.16.output.dropout' == layer_name:
            self.backbone.encoder.blocks[16].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.16.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.17' == layer_name:
            self.backbone.encoder.blocks[17]= new_layer
            self.layer_names["backbone.encoder.blocks.17"]=new_layer

        elif 'backbone.encoder.blocks.17.layernorm1' == layer_name:
            self.backbone.encoder.blocks[17].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.17.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.17.layernorm2' == layer_name:
            self.backbone.encoder.blocks[17].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.17.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.17.attention' == layer_name:
            self.backbone.encoder.blocks[17].attention= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention"]=new_layer

        elif 'backbone.encoder.blocks.17.attention.projection' == layer_name:
            self.backbone.encoder.blocks[17].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.17.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[17].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.17.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[17].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.17.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[17].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.17.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[17].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.17.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[17].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.17.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[17].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.17.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.17.output' == layer_name:
            self.backbone.encoder.blocks[17].output= new_layer
            self.layer_names["backbone.encoder.blocks.17.output"]=new_layer

        elif 'backbone.encoder.blocks.17.output.mapping' == layer_name:
            self.backbone.encoder.blocks[17].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.17.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.17.output.projection' == layer_name:
            self.backbone.encoder.blocks[17].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.17.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.17.output.dropout' == layer_name:
            self.backbone.encoder.blocks[17].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.17.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.18' == layer_name:
            self.backbone.encoder.blocks[18]= new_layer
            self.layer_names["backbone.encoder.blocks.18"]=new_layer

        elif 'backbone.encoder.blocks.18.layernorm1' == layer_name:
            self.backbone.encoder.blocks[18].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.18.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.18.layernorm2' == layer_name:
            self.backbone.encoder.blocks[18].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.18.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.18.attention' == layer_name:
            self.backbone.encoder.blocks[18].attention= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention"]=new_layer

        elif 'backbone.encoder.blocks.18.attention.projection' == layer_name:
            self.backbone.encoder.blocks[18].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.18.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[18].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.18.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[18].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.18.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[18].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.18.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[18].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.18.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[18].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.18.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[18].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.18.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.18.output' == layer_name:
            self.backbone.encoder.blocks[18].output= new_layer
            self.layer_names["backbone.encoder.blocks.18.output"]=new_layer

        elif 'backbone.encoder.blocks.18.output.mapping' == layer_name:
            self.backbone.encoder.blocks[18].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.18.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.18.output.projection' == layer_name:
            self.backbone.encoder.blocks[18].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.18.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.18.output.dropout' == layer_name:
            self.backbone.encoder.blocks[18].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.18.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.19' == layer_name:
            self.backbone.encoder.blocks[19]= new_layer
            self.layer_names["backbone.encoder.blocks.19"]=new_layer

        elif 'backbone.encoder.blocks.19.layernorm1' == layer_name:
            self.backbone.encoder.blocks[19].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.19.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.19.layernorm2' == layer_name:
            self.backbone.encoder.blocks[19].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.19.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.19.attention' == layer_name:
            self.backbone.encoder.blocks[19].attention= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention"]=new_layer

        elif 'backbone.encoder.blocks.19.attention.projection' == layer_name:
            self.backbone.encoder.blocks[19].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.19.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[19].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.19.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[19].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.19.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[19].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.19.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[19].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.19.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[19].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.19.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[19].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.19.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.19.output' == layer_name:
            self.backbone.encoder.blocks[19].output= new_layer
            self.layer_names["backbone.encoder.blocks.19.output"]=new_layer

        elif 'backbone.encoder.blocks.19.output.mapping' == layer_name:
            self.backbone.encoder.blocks[19].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.19.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.19.output.projection' == layer_name:
            self.backbone.encoder.blocks[19].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.19.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.19.output.dropout' == layer_name:
            self.backbone.encoder.blocks[19].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.19.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.20' == layer_name:
            self.backbone.encoder.blocks[20]= new_layer
            self.layer_names["backbone.encoder.blocks.20"]=new_layer

        elif 'backbone.encoder.blocks.20.layernorm1' == layer_name:
            self.backbone.encoder.blocks[20].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.20.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.20.layernorm2' == layer_name:
            self.backbone.encoder.blocks[20].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.20.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.20.attention' == layer_name:
            self.backbone.encoder.blocks[20].attention= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention"]=new_layer

        elif 'backbone.encoder.blocks.20.attention.projection' == layer_name:
            self.backbone.encoder.blocks[20].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.20.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[20].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.20.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[20].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.20.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[20].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.20.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[20].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.20.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[20].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.20.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[20].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.20.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.20.output' == layer_name:
            self.backbone.encoder.blocks[20].output= new_layer
            self.layer_names["backbone.encoder.blocks.20.output"]=new_layer

        elif 'backbone.encoder.blocks.20.output.mapping' == layer_name:
            self.backbone.encoder.blocks[20].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.20.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.20.output.projection' == layer_name:
            self.backbone.encoder.blocks[20].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.20.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.20.output.dropout' == layer_name:
            self.backbone.encoder.blocks[20].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.20.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.21' == layer_name:
            self.backbone.encoder.blocks[21]= new_layer
            self.layer_names["backbone.encoder.blocks.21"]=new_layer

        elif 'backbone.encoder.blocks.21.layernorm1' == layer_name:
            self.backbone.encoder.blocks[21].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.21.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.21.layernorm2' == layer_name:
            self.backbone.encoder.blocks[21].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.21.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.21.attention' == layer_name:
            self.backbone.encoder.blocks[21].attention= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention"]=new_layer

        elif 'backbone.encoder.blocks.21.attention.projection' == layer_name:
            self.backbone.encoder.blocks[21].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.21.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[21].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.21.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[21].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.21.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[21].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.21.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[21].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.21.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[21].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.21.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[21].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.21.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.21.output' == layer_name:
            self.backbone.encoder.blocks[21].output= new_layer
            self.layer_names["backbone.encoder.blocks.21.output"]=new_layer

        elif 'backbone.encoder.blocks.21.output.mapping' == layer_name:
            self.backbone.encoder.blocks[21].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.21.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.21.output.projection' == layer_name:
            self.backbone.encoder.blocks[21].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.21.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.21.output.dropout' == layer_name:
            self.backbone.encoder.blocks[21].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.21.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.22' == layer_name:
            self.backbone.encoder.blocks[22]= new_layer
            self.layer_names["backbone.encoder.blocks.22"]=new_layer

        elif 'backbone.encoder.blocks.22.layernorm1' == layer_name:
            self.backbone.encoder.blocks[22].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.22.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.22.layernorm2' == layer_name:
            self.backbone.encoder.blocks[22].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.22.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.22.attention' == layer_name:
            self.backbone.encoder.blocks[22].attention= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention"]=new_layer

        elif 'backbone.encoder.blocks.22.attention.projection' == layer_name:
            self.backbone.encoder.blocks[22].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.22.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[22].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.22.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[22].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.22.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[22].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.22.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[22].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.22.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[22].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.22.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[22].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.22.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.22.output' == layer_name:
            self.backbone.encoder.blocks[22].output= new_layer
            self.layer_names["backbone.encoder.blocks.22.output"]=new_layer

        elif 'backbone.encoder.blocks.22.output.mapping' == layer_name:
            self.backbone.encoder.blocks[22].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.22.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.22.output.projection' == layer_name:
            self.backbone.encoder.blocks[22].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.22.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.22.output.dropout' == layer_name:
            self.backbone.encoder.blocks[22].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.22.output.dropout"]=new_layer

        elif 'backbone.encoder.blocks.23' == layer_name:
            self.backbone.encoder.blocks[23]= new_layer
            self.layer_names["backbone.encoder.blocks.23"]=new_layer

        elif 'backbone.encoder.blocks.23.layernorm1' == layer_name:
            self.backbone.encoder.blocks[23].layernorm1= new_layer
            self.layer_names["backbone.encoder.blocks.23.layernorm1"]=new_layer

        elif 'backbone.encoder.blocks.23.layernorm2' == layer_name:
            self.backbone.encoder.blocks[23].layernorm2= new_layer
            self.layer_names["backbone.encoder.blocks.23.layernorm2"]=new_layer

        elif 'backbone.encoder.blocks.23.attention' == layer_name:
            self.backbone.encoder.blocks[23].attention= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention"]=new_layer

        elif 'backbone.encoder.blocks.23.attention.projection' == layer_name:
            self.backbone.encoder.blocks[23].attention.projection= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention.projection"]=new_layer

        elif 'backbone.encoder.blocks.23.attention.dropout' == layer_name:
            self.backbone.encoder.blocks[23].attention.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention.dropout"]=new_layer

        elif 'backbone.encoder.blocks.23.attention.prob_dropout' == layer_name:
            self.backbone.encoder.blocks[23].attention.prob_dropout= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention.prob_dropout"]=new_layer

        elif 'backbone.encoder.blocks.23.attention.softmax' == layer_name:
            self.backbone.encoder.blocks[23].attention.softmax= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention.softmax"]=new_layer

        elif 'backbone.encoder.blocks.23.attention.dense1' == layer_name:
            self.backbone.encoder.blocks[23].attention.dense1= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention.dense1"]=new_layer

        elif 'backbone.encoder.blocks.23.attention.dense2' == layer_name:
            self.backbone.encoder.blocks[23].attention.dense2= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.23.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[23].attention.dense3= new_layer
            self.layer_names["backbone.encoder.blocks.23.attention.dense3"]=new_layer

        elif 'backbone.encoder.blocks.23.output' == layer_name:
            self.backbone.encoder.blocks[23].output= new_layer
            self.layer_names["backbone.encoder.blocks.23.output"]=new_layer

        elif 'backbone.encoder.blocks.23.output.mapping' == layer_name:
            self.backbone.encoder.blocks[23].output.mapping= new_layer
            self.layer_names["backbone.encoder.blocks.23.output.mapping"]=new_layer

        elif 'backbone.encoder.blocks.23.output.projection' == layer_name:
            self.backbone.encoder.blocks[23].output.projection= new_layer
            self.layer_names["backbone.encoder.blocks.23.output.projection"]=new_layer

        elif 'backbone.encoder.blocks.23.output.dropout' == layer_name:
            self.backbone.encoder.blocks[23].output.dropout= new_layer
            self.layer_names["backbone.encoder.blocks.23.output.dropout"]=new_layer

        elif 'backbone.layernorm' == layer_name:
            self.backbone.layernorm= new_layer
            self.layer_names["backbone.layernorm"]=new_layer

        elif 'head' == layer_name:
            self.head= new_layer
            self.layer_names["head"]=new_layer


class AttentionMask(nn.Module):
    def __init__(self, seq_length):
        super(AttentionMask, self).__init__()
        self.seq_length = seq_length
        self.reshape = torch.reshape
        self.mul = torch.bmm
        ones = np.ones(shape=(seq_length, seq_length))
        self.lower_triangle_mask = torch.tensor(np.tril(ones), dtype=torch.float32).to(final_device)
        self.multiply = torch.mul

    def forward(self, input_mask):
        input_mask = torch.ne(input_mask, 0)
        input_shape = input_mask.shape
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        mask_left = self.reshape(input_mask, shape_left).to(torch.float32)
        mask_right = self.reshape(input_mask, shape_right).to(torch.float32)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = self.lower_triangle_mask.unsqueeze(0)
        # the returned shape is [bs, seq_length, seq_length]
        attention_mask = self.multiply(
            attention_mask, lower_traiangle)
        return attention_mask





def tuple_to_array(t):
    return torch.tensor(np.array(t)).to(torch.float32).to(final_device)


class LayerNorm_torch(nn.Module):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            param_init_type: The param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm_torch, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps).to(final_device)

    def forward(self, x):
        return self.layer_norm(x)


def reduce_sum(input_x, axis, skip_model=False, keep_dims=False):
    value = None
    if input_x is not None and axis is not None:
        value = input_x.cpu().detach().numpy()
        if isinstance(axis, int):
            pass
        elif axis:
            axis = tuple(set(axis))
        elif axis in ((), []) and skip_model:
            return input_x
        else:
            axis = tuple(range(len(value.shape)))
        value = np.sum(value, axis, keepdims=keep_dims)
        value = np.array(value)
        value = torch.tensor(value).to(final_device)
    return value


class Softmax(nn.Module):
    """
    Calculate the softmax results with given logits.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). The output logits of
          the backbone.

    Outputs:
        Tensor. The corresponding softmax results.
    """

    def __init__(self):
        super(Softmax, self).__init__()
        # self.softmax = nn.Softmax(dim=-1)
        self.one_hot = nn.functional.one_hot
        self.on_value = torch.tensor(1.0, dtype=torch.float32).to(final_device)
        self.off_value = torch.tensor(0.0, dtype=torch.float32).to(final_device)

    def forward(self, logits, labels):
        logits = logits.float()
        # softmax_result = self.softmax(logits)
        #
        # # Make sure labels are on the same device as logits
        # labels = labels.to(logits.device)
        # # labels = torch.abs(labels)
        # # print("labels", labels.shape)
        # labels = torch.clamp(labels, 0, logits.shape[-1] - 1)
        # one_hot_labels = self.onehot(labels, num_classes=logits.shape[-1])
        #
        # return softmax_result, one_hot_labels
        logit_max, _ = torch.max(logits, dim=-1, keepdim=True)
        # print("logit_max", logit_max.shape)
        # print("logit_max", logit_max)

        logit_sub = torch.sub(logits, logit_max)
        logit_exp = torch.exp(logit_sub)
        # print("logit_exp", logit_exp)

        exp_sum = reduce_sum(logit_exp, -1)
        exp_sum = torch.reshape(exp_sum, (exp_sum.shape[0], 1))
        # print("logit_exp", logit_exp.device)
        # print("exp_sum", exp_sum.device)
        softmax_result = torch.div(logit_exp, exp_sum.to(logit_exp.device))
        # print("softmax_result", softmax_result)

        labels = torch.clamp(labels, 0, logits.shape[-1] - 1)
        # print("labels", labels)
        # print("labels", torch.max(labels))
        # print("logits.shape[-1]", logits.shape[-1])
        # one_hot_label = self.onehot(labels, logits.shape[-1])
        # print("on_value.device", self.on_value.device)
        # print("off_value.device", self.off_value.device)
        # print("label.device", labels.device)
        # print("logits.device", logits.device)
        one_hot_label = self.one_hot(labels.to(final_device), num_classes=logits.shape[-1]) * self.on_value - self.one_hot(
            labels.to(final_device),
            num_classes=
            logits.shape[
                -1]) * \
                        self.off_value + self.off_value
        return softmax_result, one_hot_label


class NLLLoss(nn.Module):
    """
    Calculate the NLLLoss results with given softmax results and the label.

    Inputs:
        - **loss** (Tensor) - Tensor of shape (N, C). Data type is float32.

    Outputs:
        Tensor. The corresponding loss results.
    """

    def __init__(self):
        super(NLLLoss, self).__init__()
        self.eps_const = 1e-24

    def forward(self, softmax_result, one_hot_label):
        log_softmax_result = torch.log(softmax_result + self.eps_const)
        loss = log_softmax_result * one_hot_label.to(log_softmax_result.device)
        loss_unsum = -loss
        loss_reduce = torch.sum(loss_unsum, dim=-1)
        return loss_reduce

    def bprop(self, dout):
        softmax_result = dout.grad_fn.saved_tensors[0]
        one_hot_label = dout.grad_fn.saved_tensors[1]
        dlogits = softmax_result - one_hot_label
        return dlogits, torch.zeros_like(one_hot_label).to(final_device)


class CrossEntropyLoss_torch(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss_torch, self).__init__()
        self.softmax = Softmax()
        self.nllloss = NLLLoss()

    def forward(self, logits, labels, input_mask):
        """Forward process"""
        # print("logits", logits)
        # print("labels", labels)
        # print("input_mask", type(input_mask))
        st, one_hot_label = self.softmax(logits, labels)
        # st = torch.tensor(st_np).to(final_device)
        # print("st", st)
        # print("one_hot_label", one_hot_label)

        loss_reduce = self.nllloss(st, one_hot_label)
        # print("loss_reduce", loss_reduce)

        # Using input_mask to mask the loss
        input_mask = input_mask.view(-1)
        numerator = torch.sum(loss_reduce * input_mask)
        # print("numerator", numerator)
        denominator = torch.sum(input_mask)
        # print("denominator", denominator)

        # denominator = torch.sum(input_mask) + 1e-5
        loss = torch.div(numerator, denominator)
        # loss = torch.unsqueeze(loss, 0)
        return loss


import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act=nn.GELU()):
        super(FeedForward, self).__init__()
        if not (isinstance(hidden_act, str) or isinstance(hidden_act, nn.Module)):
            raise TypeError(f"For FeedForward module, the hidden_act should str type or nn.Module type, "
                            f"but got {hidden_act}.")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                             f"but got the value : {dropout_rate}.")

        self.hidden_act = hidden_act if isinstance(hidden_act, nn.Module) else getattr(F, hidden_act.lower())

        # Project to ffn_hidden_size
        self.mapping = nn.Linear(hidden_size, ffn_hidden_size).to(final_device)

        # Project back to hidden_size
        self.projection = nn.Linear(ffn_hidden_size, hidden_size).to(final_device)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """Forward process of the FeedForward"""
        x = x.float()

        # returned shape is [bs, seq_length, ffn_hidden_size] or [bs * seq_length, ffn_hidden_size]
        hidden = self.hidden_act(self.mapping(x))
        output = self.projection(hidden)
        output = self.dropout(output)
        return output


def batch_matmul(x, y, transpose_a=False, transpose_b=False):
    if transpose_a:
        x = x.transpose(-1, -2)
    if transpose_b:
        y = y.transpose(-1, -2)

    return torch.bmm(x, y)


class MultiHeadAttention(nn.Module):
    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=torch.float32,
                 softmax_compute_type=torch.float32,
                 param_init_type=torch.float32,
                 use_past=False,
                 is_parallel=True):
        super(MultiHeadAttention, self).__init__()
        if is_parallel:
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_features=hidden_size,
                                     out_features=hidden_size)
            self.projection.bias.parallel_optimizer = False
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.multiply_data = torch.tensor([
                -10000.0,
            ], dtype=softmax_compute_type).to(final_device)
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = torch.tensor(math.sqrt(math.sqrt(self.size_per_head))).to(final_device)
            self.use_past = use_past
            self.dropout = nn.Dropout(p=hidden_dropout_rate)
            self.prob_dropout = nn.Dropout(p=attention_dropout_rate)
            self.softmax = nn.Softmax()
            self.softmax_3d = nn.Softmax()

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size).to(final_device)
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size).to(final_device)
            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size).to(final_device)

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            self.ones = torch.tensor((1.0,), dtype=torch.float32).to(final_device)
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = torch.tensor(np.tile(seq_range, (batch_size, 1, 1)), torch.int64).to(final_device)
                self.seq_length = src_seq_length
                self.attention_mask = torch.tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))),
                                                   torch.int64).to(final_device)

        else:
            self.src_seq_length = src_seq_length
            self.tgt_seq_length = tgt_seq_length
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            if hidden_dropout_rate < 0 or hidden_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(hidden_dropout_rate))
            if attention_dropout_rate < 0 or attention_dropout_rate >= 1:
                raise ValueError("For 'MultiHeadAttention', the class variable 'attention_dropout_rate' must be "
                                 "in range [0, 1.0), but got the value : {}.".format(attention_dropout_rate))
            if hidden_size % num_heads != 0:
                raise ValueError("For 'MultiHeadAttention', the class variable 'hidden_size' must be a multiple "
                                 "of 'num_heads', but got the hidden_size is {} and the num_heads is {}."
                                 .format(hidden_size, num_heads))
            self.is_first_iteration = True
            # Output layer
            self.projection = Linear(in_features=hidden_size,
                                     out_features=hidden_size).to(final_device)
            self.projection.bias = Parameter(torch.ones(hidden_size)).to(final_device)
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.multiply_data = torch.tensor([
                -10000.0,
            ], dtype=softmax_compute_type).to(final_device)
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = torch.tensor(math.sqrt(math.sqrt(self.size_per_head))).to(final_device)
            self.use_past = use_past
            self.dropout = nn.Dropout(p=hidden_dropout_rate)
            self.prob_dropout = nn.Dropout(p=attention_dropout_rate)
            self.softmax = nn.Softmax()
            self.softmax_3d = nn.Softmax()

            # Query
            self.dense1 = Linear(hidden_size,
                                 hidden_size).to(final_device)
            # Key
            self.dense2 = Linear(hidden_size,
                                 hidden_size).to(final_device)

            # Value
            self.dense3 = Linear(hidden_size,
                                 hidden_size).to(final_device)

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = torch.tensor(np.tile(seq_range, (batch_size, 1, 1)), dtype=torch.int64).to(final_device)
                self.seq_length = src_seq_length
                self.attention_mask = torch.tensor(np.tril(np.ones(shape=(self.seq_length,
                                                                          self.seq_length))), dtype=torch.int64).to(final_device)
                self.not_equal = torch.not_equal
                self.reducesum = torch.sum
                self.tensor_le = torch.less_equal

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                value_past=None, batch_valid_length=None):
        """Forward process of the MultiHeadAttention"""
        ori_shape = query_tensor.shape
        # print("ori_shape", ori_shape)
        batch_size = self._get_batch_size_from_query(query_tensor)
        query_tensor, key_tensor, value_tensor = self._convert_to_2d_tensor(query_tensor,
                                                                            key_tensor,
                                                                            value_tensor)
        # print("query_tensor", query_tensor)
        # print("key_tensor", key_tensor)
        # print("value_tensor", value_tensor)
        ori_dtype = query_tensor.dtype
        query_tensor = query_tensor.to(self.dtype)
        key_tensor = key_tensor.to(self.dtype)
        value_tensor = value_tensor.to(self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # print("query1", query)
        # print("key1", key)
        # print("value1", value)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = torch.permute(
            torch.reshape(
                query,
                (batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = torch.permute(
            torch.reshape(
                key, (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                      self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = torch.permute(
            torch.reshape(
                value,
                (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # print("query2", query)
        # print("key2", key)
        # print("value2", value)
        # support input shape is [bs, seq, seq] or [bs, heads, seq, seq]
        if attention_mask is not None and len(attention_mask.shape) == 3:
            # expand attention mask from [bs, seq, seq] -> [bs, 1, seq, seq]
            attention_mask = torch.unsqueeze(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = torch.less(self.range, batch_valid_length.view(-1, 1, 1)).to(self.dtype).to(final_device)
                # Cover the key and value numbers corresponding to the padding position
                key_present = torch.mul(key, torch.unsqueeze(valid_length_vector, 2))
                value_present = torch.mul(value, torch.unsqueeze(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        # np.save("query_torch.npy", query.detach().numpy())
        # np.save("key_torch.npy", key.detach().numpy())
        # np.save("value_torch.npy", value.detach().numpy())
        # np.save("attention_mask_torch.npy", attention_mask.detach().numpy())
        attention = self._attn(query, key, value, attention_mask)
        # print("attention", attention)
        # np.save("attention_torch.npy", attention.detach().numpy())
        # exit(0)
        # Output
        output = self.projection(attention)
        # print("output1", output)
        output = self.dropout(output)
        # print("output2", output)
        output = torch.reshape(output, ori_shape)
        # print("output3", output)
        output = output.to(ori_dtype)
        # print("output4", output)
        return output, layer_present

    def _get_batch_size_from_query(self, query):
        """Get the batch size from query tensor"""
        # For the incremental prediction, the seq length for the input is 1.
        if len(query.shape) == 2 and ((self.use_past and self.is_first_iteration) or (not self.use_past)):
            return query.shape[0] // self.src_seq_length
        return query.shape[0]

    def _get_seq_length_under_incremental(self, length):
        r"""Return the length of the tensor.
            For the incremental prediction, the seq length for the input is 1.
        """
        if self.use_past and not self.is_first_iteration:
            return 1
        return length

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor):
        """convert a nd tensor to a 2d tensor"""
        query_tensor = query_tensor.view(-1, query_tensor.shape[-1])
        key_tensor = key_tensor.view(-1, key_tensor.shape[-1])
        value_tensor = value_tensor.view(-1, value_tensor.shape[-1])
        return query_tensor, key_tensor, value_tensor

    def _merge_heads(self, x):
        x = torch.permute(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = torch.reshape(x, new_shape)
        return x_merge

    def _softmax(self, attention_scores):
        # print("attention_scores000", attention_scores)
        # if self.softmax_dtype == torch.float32:
        # not here
        attention_probs = self.softmax(attention_scores)

        # else:
        #     shape = attention_scores.shape
        #     print("softmax_shape", shape)
        #     # attention probs
        #     attention_probs = self.softmax_3d(
        #         torch.reshape(attention_scores, (shape[0], -1, shape[-1])))
        #     attention_probs = torch.reshape(attention_probs, shape)
        return attention_probs

    def _attn(self, query, key, value, attention_mask):
        # print("value", value)
        factor = self.scale_factor.to(query.dtype)
        # np.save("factor_torch.npy", factor.detach().cpu().numpy())
        dev = query.device
        factor = factor.to(dev)
        # print("factor", factor)
        query = torch.div(query, factor)
        key = torch.div(key, factor)
        # print("query", query)
        # print("key", key)
        score = torch.matmul(query, key)
        # print("score", score)

        ori_dtype = score.dtype
        attention_scores = score.to(self.softmax_dtype)
        # np.save("attention_scores_torch.npy", attention_scores.detach().cpu().numpy())
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = (key[..., :query.shape[0], 0, 0, 0] != 0).float().sum(dim=(1, 2, 3))
                # Get the precise position index
                index = torch.sub(current_index.to(torch.int64), 1)
                index = torch.reshape(index, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = self.tensor_le(self.range, index).to(torch.int64)
                attention_mask = torch.unsqueeze(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = torch.sub(
                self.ones.to(attention_scores.dtype).to(final_device),
                attention_mask.to(attention_scores.dtype).to(final_device))

            adder = torch.mul(multiplu_out, self.multiply_data)
            dev = attention_scores.device
            adder = adder.to(dev)
            attention_scores = torch.add(adder, attention_scores)
        # print("attention_scores", attention_scores)
        # attention probs
        # np.save("attention_scores.npy", attention_scores.detach().cpu().numpy())
        attention_probs = self._softmax(attention_scores)
        # np.save("attention_probs.npy", attention_probs.detach().cpu().numpy())
        # print("attention_probs1", attention_probs)
        attention_probs = attention_probs.to(ori_dtype)
        # print("attention_probs2", attention_probs)
        attention_probs = self.prob_dropout(attention_probs)
        # print("attention_probs3", attention_probs)
        # np.save("attention_probs.npy", attention_probs.detach().cpu().numpy())
        # np.save("value.npy", value.detach().cpu().numpy())
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        np.save("value_torch.npy", value.detach().cpu().numpy())
        np.save("attention_probs_torch.npy", attention_probs.detach().cpu().numpy())
        weighted_values = torch.matmul(attention_probs, value)
        np.save("weighted_values_torch.npy", weighted_values.detach().cpu().numpy())

        # print("weighted_values", weighted_values)
        attention_merge = self._merge_heads(weighted_values)
        # np.save("attention_merge_torch.npy", attention_merge.detach().cpu().numpy())
        # print("attention_merge", attention_merge)
        # exit(0)
        return attention_merge


class TransformerEncoderLayer_torch(nn.Module):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=torch.float32,
                 softmax_compute_type=torch.float32,
                 param_init_type=torch.float32,
                 hidden_act=nn.GELU(),
                 use_past=False):
        super(TransformerEncoderLayer_torch, self).__init__()
        self.batch_size = batch_size
        self.use_moe = False
        self.use_past = use_past
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layernorm1 = nn.LayerNorm((hidden_size,))
        self.layernorm2 = nn.LayerNorm((hidden_size,))

        self.attention = MultiHeadAttention(batch_size=batch_size,
                                            src_seq_length=seq_length,
                                            tgt_seq_length=seq_length,
                                            hidden_size=hidden_size,
                                            num_heads=num_heads,
                                            hidden_dropout_rate=hidden_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            softmax_compute_type=softmax_compute_type,
                                            param_init_type=param_init_type,
                                            use_past=use_past)
        self.output = FeedForward(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size)
        self.post_layernorm_residual = post_layernorm_residual
        self.add = torch.add
        self.add_3d = torch.add
        self.dtype = torch.float32
        self.key_past = None
        self.value_past = None

        if self.use_past:
            # operator used for state reuse
            self.reducesum = torch.sum
            self.not_equal = torch.not_equal
            size_per_head = hidden_size // num_heads
            self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
            self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
            # parameters saving key and value states
            # self.key_past = nn.Parameter(torch.tensor(np.zeros(shape=self.key_shape), self.dtype).to(final_device))
            # self.value_past = nn.Parameter(torch.tensor(np.zeros(shape=self.value_shape), self.dtype).to(final_device))
            self.tile = torch.tile
            self.mul = torch.mul
            # self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))

    def forward(self, x, input_mask=None, batch_valid_length=None):
        """forward process"""
        x_shape = x.shape
        x = torch.reshape(x, (-1, x_shape[-1]))
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = input_x.to(self.dtype)

        attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = output_x.to(self.dtype)
        mlp_logit = self.output(output_x)

        if len(x_shape) == 3:
            output_x = torch.reshape(output_x, x_shape)
            mlp_logit = torch.reshape(mlp_logit, x_shape)
            x = torch.reshape(x, x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
                output = torch.reshape(output, (-1, x_shape[-1]))
                output = self.layernorm1(output)
                output = torch.reshape(output, x_shape)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
                output = self.layernorm1(output)
            else:
                output = self.add(x, mlp_logit)
            output = torch.reshape(output, x_shape)

        return output, layer_present


class TransformerEncoder_torch(nn.Module):
    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act=nn.GELU(),
                 post_layernorm_residual=False,
                 layernorm_compute_type=torch.float32,
                 softmax_compute_type=torch.float32,
                 param_init_type=torch.float32,
                 lambda_func=None,
                 offset=0,
                 use_past=False,
                 # moe_config=default_moe_config,
                 # parallel_config=default_transformer_config
                 is_parallel=True
                 ):
        super(TransformerEncoder_torch, self).__init__()
        # _check_config(parallel_config)
        # _check_moe_config(moe_config, parallel_config)
        self.use_moe = False
        # if batch_size or use_past:
        #     Validator.check_positive_int(batch_size)
        # config_to_layer = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        if is_parallel:
            self.add = torch.add
            self.aux_loss = torch.tensor(0.0, dtype=torch.float32).to(final_device)
            self.num_layers = num_layers
            self.blocks = nn.ModuleList()
            for i in range(num_layers):
                block = TransformerEncoderLayer_torch(hidden_size=hidden_size,
                                                      batch_size=batch_size,
                                                      ffn_hidden_size=ffn_hidden_size,
                                                      seq_length=seq_length,
                                                      attention_dropout_rate=attention_dropout_rate,
                                                      hidden_dropout_rate=hidden_dropout_rate,
                                                      layernorm_compute_type=layernorm_compute_type,
                                                      softmax_compute_type=softmax_compute_type,
                                                      num_heads=num_heads,
                                                      hidden_act=hidden_act,
                                                      post_layernorm_residual=post_layernorm_residual,
                                                      param_init_type=param_init_type,
                                                      use_past=use_past)
                # If the user doesn't pass the fusion function, use the default one
                # if not lambda_func:
                #     lambda_func = _get_lambda_func()
                #
                # lambda_func(block, layer_id=i, layers=num_layers,
                #             offset=offset, parallel_config=parallel_config)
                self.blocks.append(block)

    def forward(self, hidden_states, attention_mask):
        """forward process"""
        present_layer = ()
        if self.use_moe:
            accum_loss = self.aux_loss
            for i in range(self.num_layers):
                hidden_states, present, aux_loss = self.blocks[i](hidden_states,
                                                                  attention_mask)
                present_layer = present_layer + (present,)
                accum_loss = self.add(accum_loss, aux_loss)
            return hidden_states, present_layer, accum_loss

        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask)
            present_layer = present_layer + (present,)

        return hidden_states, present_layer


class GPT_Model(nn.Module):
    def __init__(self, config):
        super(GPT_Model, self).__init__()
        self.get_attention_mask = AttentionMask(seq_length=config.seq_length)
        self.word_embedding = EmbeddingLookupPyTorch(config)
        self.position_embedding = nn.Embedding(config.seq_length, config.embedding_size)
        # self.blocks = nn.ModuleList()
        self.encoder = TransformerEncoder_torch(batch_size=config.batch_size,
                                                num_layers=config.num_layers,
                                                hidden_size=config.embedding_size,
                                                ffn_hidden_size=config.embedding_size * 4,
                                                seq_length=config.seq_length,
                                                num_heads=config.num_heads, )
        self.layernorm = LayerNorm([config.embedding_size])
        self.use_past = config.use_past
        self.past = tuple([None] * config.num_layers)
        self.num_layers = config.num_layers

    def forward(self, input_ids, input_mask, layer_past=None):
        input_embedding, embedding_table = self.word_embedding(input_ids)
        batch_size, seq_length = input_ids.shape
        input_position = tuple_to_array(torch.arange(seq_length))
        input_position = torch.tile(input_position, (batch_size, 1)).long()
        input_position = torch.clamp(input_position, max=511)

        position_embedding = self.position_embedding(input_position)
        hidden_states = input_embedding + position_embedding

        hidden_states = hidden_states.to(torch.float32)
        attention_mask = self.get_attention_mask(input_mask)

        hidden_states, present_layer = self.encoder(hidden_states, attention_mask)
        output_state = self.layernorm(hidden_states)
        return output_state, present_layer, embedding_table


# GPTWithLoss translation
class GPTWithLoss(nn.Module):
    def __init__(self, gpt_model, loss_fn, eos_token=50256):
        super(GPTWithLoss, self).__init__()
        self.gpt_model = gpt_model
        self.loss_fn = loss_fn
        self.eos_token = eos_token

    def forward(self, input_ids):
        logits = self.gpt_model(input_ids)
        labels = input_ids[:, 1:]
        # logits = logits[:, :-1, :]
        # logits = logits.contiguous().view(-1, logits.size(-1))
        labels = labels.contiguous().view(-1)
        mask = (labels != self.eos_token).float()
        losses = self.loss_fn(logits, labels)
        masked_losses = losses * mask
        loss = masked_losses.sum() / mask.sum()
        return loss


# EvalNet translation
class EvalNetPyTorch(nn.Module):
    def __init__(self, backbone, generate=False):
        super(EvalNetPyTorch, self).__init__()
        self.backbone = backbone
        self.generate = generate
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids):
        logits = self.backbone(input_ids)
        if self.generate:
            outputs = self.log_softmax(logits)
            outputs = torch.exp(outputs)
        else:
            outputs = torch.argmax(logits, dim=-1)
        return outputs


if __name__ == '__main__':
    config = GPTConfig(batch_size=1,
                       seq_length=1024,
                       vocab_size=50257,
                       embedding_size=1024,
                       num_layers=24,
                       num_heads=16,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.1,
                       compute_dtype=torch.float32,
                       use_past=False)
    model = GPTPyTorch(config).to(final_device)
    # loss = CrossEntropyLoss()
    # model = GPTWithLoss(model, loss, eos_token=0)
    np_data = [np.ones((1, 1025))]
    # dtypes = [torch.int64]
    # input_data = torchinfoplus.np_2_tensor(np_data, dtypes, device=device)
    # res, global_layer_info = torchinfoplus.summary(
    #     model=model,
    #     input_data=input_data,
    #     dtypes=dtypes,
    #     col_names=['input_size', 'output_size', 'name'],
    #     verbose=2,
    #     depth=10)
    # input_datas = torchinfoplus.get_input_datas(global_layer_info)
    a = torch.tensor(np_data[0], dtype=torch.int64).to(final_device)
    # tokens = a[:, :-1]
    # input_mask = torch.ne(tokens, 50256).to(torch.float32)
    print(model(a).shape)
