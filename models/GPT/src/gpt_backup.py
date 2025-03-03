# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""GPT model"""
import math
import mindspore
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
import torch
from mindspore import Tensor, ops
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import LayerNorm
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.parallel._transformer.layers import _LayerNorm
from mindspore.parallel._transformer.transformer import AttentionMask
import troubleshooter as ts
ts.widget.fix_random(0)
from comparer import compare_models_new
from gpt_torch import GPTPyTorch
from src.utils import GPTConfig


class EmbeddingLookup(nn.Cell):
    """
    The embedding lookup table for vocabulary

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32

    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size, seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.embedding_table = Parameter(initializer(TruncatedNormal(0.02), [self.vocab_size, self.embedding_size]))
        self.gather = P.Gather()
        self.shape = (-1, config.seq_length, config.embedding_size)

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table.value()


class FeedForward(nn.Cell):
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act=nn.GELU()):
        super(FeedForward, self).__init__()
        if not (isinstance(hidden_act, str) or isinstance(hidden_act, nn.Cell)):
            raise TypeError(f"For FeedForward module, the hidden_act should str type or nn.Module type, "
                            f"but got {hidden_act}.")
        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("For 'FeedForward', the class variable 'dropout_rate' must be in the range [0, 1.0), "
                             f"but got the value : {dropout_rate}.")
        self.hidden_act = hidden_act if isinstance(hidden_act, nn.Cell) else getattr(F, hidden_act.lower())
        # Project to ffn_hidden_size
        self.mapping = nn.Dense(hidden_size, ffn_hidden_size)
        # Project back to hidden_size
        self.projection = nn.Dense(ffn_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def construct(self, x):
        """Forward process of the FeedForward"""
        x = x.float()
        hidden = self.hidden_act(self.mapping(x))
        output = self.projection(hidden)
        output = self.dropout(output)
        return output


class MultiHeadAttention(nn.Cell):
    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mindspore.float32,
                 softmax_compute_type=mindspore.float32,
                 param_init_type=mindspore.float32,
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
            self.projection = nn.Dense(in_channels=hidden_size,
                                       out_channels=hidden_size)
            self.projection.bias = Parameter(initializer("ones", [hidden_size], param_init_type))
            # self.projection.bias.parallel_optimizer = False
            self.n_head = num_heads
            # embedding size per head
            self.size_per_head = hidden_size // self.n_head
            self.multiply_data = mindspore.Tensor([
                -10000.0,
            ], dtype=softmax_compute_type)
            # Normalize factor for attention, sqrt(dk) as widely used
            self.scale_factor = mindspore.Tensor(math.sqrt(math.sqrt(self.size_per_head)))
            self.use_past = use_past
            self.dropout = nn.Dropout(p=hidden_dropout_rate)
            self.prob_dropout = nn.Dropout(p=attention_dropout_rate)
            self.softmax = nn.Softmax()
            self.softmax_3d = nn.Softmax()

            # Query
            self.dense1 = nn.Dense(hidden_size,
                                   hidden_size)
            # Key
            self.dense2 = nn.Dense(hidden_size,
                                   hidden_size)
            # Value
            self.dense3 = nn.Dense(hidden_size,
                                   hidden_size)

            self.dtype = compute_dtype
            self.softmax_dtype = softmax_compute_type
            self.ones = mindspore.Tensor((1.0,), dtype=mindspore.float32)
            if self.use_past:
                # operators used for state reuse
                seq_range = np.arange(src_seq_length).reshape(1, 1, -1)
                self.range = mindspore.Tensor(np.tile(seq_range, (batch_size, 1, 1)), mindspore.int32)
                self.seq_length = src_seq_length
                self.attention_mask = mindspore.Tensor(np.tril(np.ones(shape=(self.seq_length, self.seq_length))),
                                                       mindspore.int32)

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None):
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
        query_tensor = query_tensor.astype(self.dtype)
        key_tensor = key_tensor.astype(self.dtype)
        value_tensor = value_tensor.astype(self.dtype)
        # multi head attention: query, key, value are derived from the same inputs
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        # print("query1", query)
        # print("key1", key)
        # print("value1", value)
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        query = mindspore.ops.permute(
            mindspore.ops.reshape(
                query,
                (batch_size, self._get_seq_length_under_incremental(self.src_seq_length),
                 self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        # the returned shape is [bs, size_per_head, seq_length, num_heads]
        key = mindspore.ops.permute(
            mindspore.ops.reshape(
                key, (batch_size, self._get_seq_length_under_incremental(self.tgt_seq_length),
                      self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        # the returned shape is [bs, num_heads, seq_length, size_per_head]
        value = mindspore.ops.permute(
            mindspore.ops.reshape(
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
            attention_mask = mindspore.ops.unsqueeze(attention_mask, 1)
        # key and value for current token(s)
        key_present = key
        value_present = value
        if self.use_past:
            # The first graph with the input size of (bs, seq_length)
            if self.is_first_iteration:
                # Get the valid input length without padding
                valid_length_vector = mindspore.ops.less(self.range, batch_valid_length.view(-1, 1, 1)).astype(self.dtype)
                # Cover the key and value numbers corresponding to the padding position
                key_present = mindspore.ops.mul(key, mindspore.ops.unsqueeze(valid_length_vector, 2))
                value_present = mindspore.ops.mul(value, mindspore.ops.unsqueeze(valid_length_vector, 3))
            # The second graph with the inpus size of (bs, 1)
            # the shape of query is (bs, num_heads, 1, size_per_head)
            # the shape of key is   (bs, num_heads, size_per_head, 1)
            # the shape of value is (bs, num_heads, 1, size_per_head)

        layer_present = (key_present, value_present)
        # multi head attention considering attention mask
        # the return shape is [bs * seq_length, hidden_size]
        # np.save("query.npy", query.asnumpy())
        # np.save("key.npy", key.asnumpy())
        # np.save("value.npy", value.asnumpy())
        # np.save("attention_mask.npy", attention_mask.asnumpy())
        attention = self._attn(query, key, value, attention_mask)
        # print("attention", attention)
        # Output
        # np.save("attention.npy", attention.asnumpy())
        # exit(0)
        output = self.projection(attention)
        # print("output1", output)
        output = self.dropout(output)
        # print("output2", output)
        output = mindspore.ops.reshape(output, ori_shape)
        # print("output3", output)
        output = output.astype(ori_dtype)
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
        x = mindspore.ops.permute(
            x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = mindspore.ops.reshape(x, new_shape)
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
        factor = self.scale_factor.astype(query.dtype)
        # np.save("factor.npy", factor.asnumpy())
        query = mindspore.ops.div(query, factor)
        key = mindspore.ops.div(key, factor)
        score = mindspore.ops.matmul(query, key)
        ori_dtype = score.dtype
        attention_scores = score.to(self.softmax_dtype)
        # np.save("attention_scores.npy", attention_scores.asnumpy())
        if attention_mask is not None:
            if self.use_past and not self.is_first_iteration:
                # Calculate the current total token
                current_index = (key[..., :query.shape[0], 0, 0, 0] != 0).float().sum(dim=(1, 2, 3))
                # Get the precise position index
                index = mindspore.ops.sub(current_index.astype(mindspore.int32), 1)
                index = mindspore.ops.reshape(index, (-1, 1, 1))
                # Calculate the attention_mask matrix via the position index
                attention_mask = self.tensor_le(self.range, index).astype(mindspore.int32)
                attention_mask = mindspore.ops.unsqueeze(attention_mask, 2)
            # Minus 10000 for the position where masked to exclude them from softmax
            multiplu_out = mindspore.ops.sub(
                self.ones.to(attention_scores.dtype),
                attention_mask.to(attention_scores.dtype))

            adder = mindspore.ops.mul(multiplu_out, self.multiply_data)
            attention_scores = mindspore.ops.add(adder, attention_scores)
        # print("attention_scores", attention_scores)
        # attention probs
        # np.save("attention_scores.npy", attention_scores.detach().cpu().numpy())
        attention_probs = self._softmax(attention_scores)
        # np.save("attention_probs.npy", attention_probs.detach().cpu().numpy())
        # print("attention_probs1", attention_probs)
        attention_probs = attention_probs.astype(ori_dtype)
        # print("attention_probs2", attention_probs)
        attention_probs = self.prob_dropout(attention_probs)
        # print("attention_probs3", attention_probs)
        # np.save("attention_probs.npy", attention_probs.detach().cpu().numpy())
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        # np.save("value.npy", value.asnumpy())
        # np.save("attention_probs.npy", attention_probs.asnumpy())
        weighted_values = mindspore.ops.matmul(attention_probs, value)

        # np.save("weighted_values.npy", weighted_values.asnumpy())
        # np.save("RESULT.npy", weighted_values.detach().cpu().numpy())

        # print("weighted_values", weighted_values)
        attention_merge = self._merge_heads(weighted_values)
        # np.save("attention_merge.npy", attention_merge.asnumpy())
        # print("attention_merge", attention_merge)
        # exit(0)
        return attention_merge


class TransformerEncoder(nn.Cell):
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
                 layernorm_compute_type=mindspore.float32,
                 softmax_compute_type=mindspore.float32,
                 param_init_type=mindspore.float32,
                 use_past=False,
                 # moe_config=default_moe_config,
                 # parallel_config=default_transformer_config
                 is_parallel=True
                 ):
        super(TransformerEncoder, self).__init__()
        # _check_config(parallel_config)
        # _check_moe_config(moe_config, parallel_config)
        self.use_moe = False
        # if batch_size or use_past:
        #     Validator.check_positive_int(batch_size)
        # config_to_layer = parallel_config.moe_parallel_config if self.use_moe else parallel_config.dp_mp_config
        if is_parallel:
            self.add = ops.add
            self.aux_loss = mindspore.Tensor(0.0, dtype=mindspore.float32)
            self.num_layers = num_layers
            self.blocks = nn.CellList()
            for i in range(num_layers):
                block = TransformerEncoderLayer(hidden_size=hidden_size,
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

    def construct(self, hidden_states, attention_mask):
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


class TransformerEncoderLayer(nn.Cell):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 use_past=False,
                 is_parallel=True):
        super(TransformerEncoderLayer, self).__init__()
        self.batch_size = batch_size
        if is_parallel:
            self.use_past = use_past
            self.seq_length = seq_length
            self.hidden_size = hidden_size
            self.layernorm1 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
            self.layernorm2 = LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
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
            # Feed Forward Network, FFN
            self.output = FeedForward(hidden_size=hidden_size,
                                      dropout_rate=hidden_dropout_rate,
                                      ffn_hidden_size=ffn_hidden_size)
            self.post_layernorm_residual = post_layernorm_residual
            self.add = P.Add()
            self.add_3d = P.Add()
            self.dtype = mstype.float16
            self.key_past = None
            self.value_past = None

            if self.use_past:
                # operator used for state reuse
                self.reducesum = P.ReduceSum()
                self.not_equal = P.NotEqual()
                self.slice = P.StridedSlice()
                size_per_head = hidden_size // num_heads
                self.key_shape = (batch_size, num_heads, size_per_head, seq_length)
                self.value_shape = (batch_size, num_heads, seq_length, size_per_head)
                # parameters saving key and value states
                self.key_past = Parameter(Tensor(np.zeros(shape=self.key_shape), self.dtype), name="key_past")
                self.value_past = Parameter(Tensor(np.zeros(shape=self.value_shape), self.dtype), name="value_past")
                self.tile = P.Tile()
                self.mul = P.Mul()
                self.assign = P.Assign()

    def construct(self, x, input_mask=None, init_reset=True, batch_valid_length=None):
        """forward process"""
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        aux_loss = None
        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            x = P.Reshape()(x, x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
                output = F.reshape(output, (-1, x_shape[-1]))
                output = self.layernorm1(output)
                output = F.reshape(output, x_shape)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
                output = self.layernorm1(output)
            else:
                output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)

        return output, layer_present


class GPT_Model(nn.Cell):
    """
    The backbone of GPT network

    Args:
        config(GPTConfig): the config of network

    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map

    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(GPT_Model, self).__init__()
        self.get_attention_mask = AttentionMask(seq_length=config.seq_length)
        self.word_embedding = EmbeddingLookup(config)
        self.position_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                               embedding_table=TruncatedNormal(0.02))
        self.blocks = nn.CellList()
        self.encoder = TransformerEncoder(batch_size=config.batch_size,
                                          num_layers=config.num_layers,
                                          hidden_size=config.embedding_size,
                                          ffn_hidden_size=config.embedding_size * 4,
                                          seq_length=config.seq_length,
                                          num_heads=config.num_heads, )
        self.layernorm = _LayerNorm((config.embedding_size,)).to_float(config.compute_dtype)
        self.use_past = config.use_past
        self.past = tuple([None] * config.num_layers)
        self.num_layers = config.num_layers

    def construct(self, input_ids, input_mask, layer_past=None):
        """GPT model"""
        input_embedding, embedding_table = self.word_embedding(input_ids)
        batch_size, seq_length = F.shape(input_ids)
        input_position = F.tuple_to_array(F.make_range(seq_length))
        input_position = P.Tile()(input_position, (batch_size, 1))
        position_embedding = self.position_embedding(input_position)
        hidden_states = input_embedding + position_embedding

        hidden_states = P.Cast()(hidden_states, mstype.float16)
        attention_mask = self.get_attention_mask(input_mask)

        hidden_states, present_layer = self.encoder(hidden_states, attention_mask)
        output_state = self.layernorm(hidden_states)
        return output_state, present_layer, embedding_table


class GPT_Head(nn.Cell):
    """
    Head for GPT to get the logits of each token in the vocab

    Args:
        config(GPTConfig): the config of network

    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary

    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """

    def __init__(self, config):
        super(GPT_Head, self).__init__()
        self.matmul = P.MatMul(transpose_b=True)
        self.embedding_size = config.embedding_size
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        state = P.Reshape()(state, (-1, self.embedding_size))
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits


config = GPTConfig(batch_size=4,
                   seq_length=1024,
                   vocab_size=50257,
                   embedding_size=1024,
                   num_layers=24,
                   num_heads=16,
                   expand_ratio=4,
                   post_layernorm_residual=False,
                   dropout_rate=0.1,
                   compute_dtype=mstype.float16,
                   use_past=False)


class GPT(nn.Cell):
    def __init__(self):
        super(GPT, self).__init__()
        self.config = config
        self.backbone = GPT_Model(self.config)
        self.head = GPT_Head(self.config)
        # 在BERT模型中，通常使用一个大小为 50256 的词汇表
        self.eos_token = 50256

    def construct(self, input_ids):
        tokens = input_ids[:, :-1]
        input_mask = F.cast(F.not_equal(tokens, self.eos_token), mstype.float32)
        output_states, _, embedding_table = self.backbone(tokens, input_mask)
        # print("output_states", output_states.shape)
        # print("embedding_table", embedding_table.shape)
        logits = self.head(output_states, embedding_table)
        return logits
        # return output_states


class GPTWithLoss(nn.Cell):
    """
    GPT training loss

    Args:
        network: backbone network of GPT2/3
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token

    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map

    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, network, loss, eos_token=50256):
        super(GPTWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token

    def construct(self, input_ids, past=None):
        tokens = input_ids[:, :-1]
        input_mask = F.cast(F.not_equal(tokens, self.eos_token), mstype.float32)
        logits = self.network(input_ids)
        labels = input_ids[:, 1:]
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output


class EvalNet(nn.Cell):
    """
    GPT evaluation net

    Args:
        backbone: backbone network of GPT2/3
        generate: enable generate mode

    Inputs:
        input_ids: the tokenized inpus

    Returns:
        outputs: Tensor, corresponding output for different tasks
    """

    def __init__(self, backbone, generate=False):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.Argmax()
        self.generate = generate
        self.cast = P.Cast()

    def construct(self, input_ids):
        """evaluation net"""
        # input_mask = self.cast(input_mask, mstype.float32)
        logits = self.backbone(input_ids)
        if self.generate:
            outputs = nn.LogSoftmax()(logits)
            outputs = F.tensor_pow(np.e, outputs)
        else:
            outputs = self.argmax(logits)
        return outputs


if __name__ == '__main__':
    from gpt_torch import device
    mindspore.set_context(pynative_synchronize=True)
    model = GPT()
    # mindspore.load_checkpoint('./convert_ms.ckpt', model)
    model.set_train(False)
    model_torch = GPTPyTorch(config).to(device)
    # model_torch.load_state_dict(torch.load('./torch_net.path'))
    model_torch.eval()
    # loss = CrossEntropyLoss()
    # model = GPTWithLoss(model, loss)
    np_data = [np.ones((1, 1025))]
    dtypes = [mindspore.int32]
    # input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
    # res, global_layer_info = mindsporeinfoplus.summary_plus(
    #     model=model,
    #     input_data=input_data,
    #     dtypes=dtypes,
    #     col_names=['input_size', 'output_size', 'name'],
    #     verbose=2,
    #     depth=10)
    # input_datas = mindsporeinfoplus.get_input_datas(global_layer_info)
    anp = (mindspore.numpy.ones((1, 1025)).astype(mindspore.int64),)
    diff_finder = ts.migrator.NetDifferenceFinder(pt_net=model_torch, ms_net=model, fix_seed=0, auto_conv_ckpt=2)  #
    ts.migrator.get_weight_map(pt_net=model_torch, weight_map_save_path='./torch_net_map.json', print_map=False)

    torch.save(model_torch.state_dict(), './torch_net.path')

    ts.migrator.convert_weight(weight_map_path='./torch_net_map.json', pt_file_path='./torch_net.path',
                               ms_file_save_path='./convert_ms.ckpt', print_conv_info=False)

    param_dict = mindspore.load_checkpoint('./convert_ms.ckpt')
    mindspore.load_param_into_net(model, param_dict)
    res, _, _ = compare_models_new(model, model_torch, np_data, dtypes, [torch.int64])
    print("res", res)
    # a = mindspore.Tensor(np_data[0], dtype=mindspore.int32)
    # tokens = a[:, :-1]
    # input_mask = F.cast(F.not_equal(tokens, 50256), mstype.float32)
    # res = model(tokens, input_mask)
    # res = model(a)
    # print(res.shape)
    # print(res[0].shape)
    # print(res[1][0].shape)
    # print(res[2].shape)
