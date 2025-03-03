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

# from comparer import compare_models_new
# from gpt_torch import GPTPyTorch
from network.nlp.GPT.src.utils import GPTConfig


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
                valid_length_vector = mindspore.ops.less(self.range, batch_valid_length.view(-1, 1, 1)).astype(
                    self.dtype)
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
        # np.save("value.npy", value.detach().cpu().numpy())
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
            self.dtype = mstype.float32
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

        hidden_states = P.Cast()(hidden_states, mstype.float32)
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
                   num_layers=8,
                   num_heads=16,
                   expand_ratio=4,
                   post_layernorm_residual=False,
                   dropout_rate=0.1,
                   compute_dtype=mstype.float32,
                   use_past=False)


class GPT(nn.Cell):
    def __init__(self):
        super(GPT, self).__init__()
        self.config = config
        self.backbone = GPT_Model(self.config)
        self.head = GPT_Head(self.config)
        # 在BERT模型中，通常使用一个大小为 50256 的词汇表
        self.eos_token = 50256

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
            'backbone.word_embedding': [1, 1024, 1024],#delete the second output
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
                                         'backbone.layernorm','head']],
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
            'backbone.get_attention_mask': [mindspore.float32],
            'backbone.word_embedding': [mindspore.int32],
            'backbone.position_embedding': [mindspore.int32],
            'backbone.encoder.blocks.0.layernorm1': [mindspore.float16],
            'backbone.encoder.blocks.0.layernorm2': [mindspore.float16],
            'backbone.encoder.blocks.0.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.0.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.0.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.0.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.0.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.0.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.0.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.0.output.hidden_act': [mindspore.float32],
            'backbone.encoder.blocks.0.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.0.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.0.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.1.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.1.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.1.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.1.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.1.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.1.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.1.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.1.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.1.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.1.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.1.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.1.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.2.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.2.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.2.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.2.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.2.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.2.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.2.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.2.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.2.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.2.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.2.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.2.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.3.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.3.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.3.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.3.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.3.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.3.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.3.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.3.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.3.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.3.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.3.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.3.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.4.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.4.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.4.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.4.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.4.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.4.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.4.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.4.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.4.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.4.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.4.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.4.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.5.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.5.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.5.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.5.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.5.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.5.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.5.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.5.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.5.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.5.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.5.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.5.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.6.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.6.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.6.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.6.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.6.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.6.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.6.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.6.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.6.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.6.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.6.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.6.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.7.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.7.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.7.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.7.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.7.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.7.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.7.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.7.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.7.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.7.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.7.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.7.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.8.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.8.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.8.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.8.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.8.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.8.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.8.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.8.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.8.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.8.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.8.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.8.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.9.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.9.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.9.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.9.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.9.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.9.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.9.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.9.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.9.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.9.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.9.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.9.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.10.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.10.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.10.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.10.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.10.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.10.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.10.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.10.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.10.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.10.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.10.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.10.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.11.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.11.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.11.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.11.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.11.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.11.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.11.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.11.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.11.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.11.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.11.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.11.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.12.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.12.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.12.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.12.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.12.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.12.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.12.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.12.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.12.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.12.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.12.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.12.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.13.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.13.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.13.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.13.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.13.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.13.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.13.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.13.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.13.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.13.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.13.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.13.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.14.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.14.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.14.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.14.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.14.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.14.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.14.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.14.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.14.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.14.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.14.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.14.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.15.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.15.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.15.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.15.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.15.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.15.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.15.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.15.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.15.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.15.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.15.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.15.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.16.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.16.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.16.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.16.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.16.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.16.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.16.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.16.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.16.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.16.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.16.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.16.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.17.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.17.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.17.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.17.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.17.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.17.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.17.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.17.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.17.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.17.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.17.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.17.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.18.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.18.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.18.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.18.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.18.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.18.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.18.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.18.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.18.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.18.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.18.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.18.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.19.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.19.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.19.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.19.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.19.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.19.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.19.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.19.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.19.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.19.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.19.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.19.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.20.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.20.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.20.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.20.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.20.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.20.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.20.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.20.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.20.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.20.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.20.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.20.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.21.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.21.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.21.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.21.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.21.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.21.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.21.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.21.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.21.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.21.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.21.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.21.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.22.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.22.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.22.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.22.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.22.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.22.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.22.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.22.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.22.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.22.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.22.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.22.output.dropout': [mindspore.float32],
            'backbone.encoder.blocks.23.layernorm1': [mindspore.float32],
            'backbone.encoder.blocks.23.layernorm2': [mindspore.float32],
            'backbone.encoder.blocks.23.attention.projection': [mindspore.float32],
            'backbone.encoder.blocks.23.attention.dropout': [mindspore.float32],
            'backbone.encoder.blocks.23.attention.prob_dropout': [mindspore.float32],
            'backbone.encoder.blocks.23.attention.softmax': [mindspore.float32],
            'backbone.encoder.blocks.23.attention.dense1': [mindspore.float32],
            'backbone.encoder.blocks.23.attention.dense2': [mindspore.float32],
            'backbone.encoder.blocks.23.attention.dense3': [mindspore.float32],
            'backbone.encoder.blocks.23.output.mapping': [mindspore.float32],
            'backbone.encoder.blocks.23.output.projection': [mindspore.float32],
            'backbone.encoder.blocks.23.output.dropout': [mindspore.float32],
            'backbone.layernorm': [mindspore.float32],
            'head': [mindspore.float16, mindspore.float32]
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
        #     "backbone.encoder.blocks.8": self.backbone.encoder.blocks[8],
        #     "backbone.encoder.blocks.8.layernorm1": self.backbone.encoder.blocks[8].layernorm1,
        #     "backbone.encoder.blocks.8.layernorm2": self.backbone.encoder.blocks[8].layernorm2,
        #     "backbone.encoder.blocks.8.attention": self.backbone.encoder.blocks[8].attention,
        #     "backbone.encoder.blocks.8.attention.projection": self.backbone.encoder.blocks[8].attention.projection,
        #     "backbone.encoder.blocks.8.attention.dropout": self.backbone.encoder.blocks[8].attention.dropout,
        #     "backbone.encoder.blocks.8.attention.prob_dropout": self.backbone.encoder.blocks[8].attention.prob_dropout,
        #     "backbone.encoder.blocks.8.attention.softmax": self.backbone.encoder.blocks[8].attention.softmax,
        #     "backbone.encoder.blocks.8.attention.dense1": self.backbone.encoder.blocks[8].attention.dense1,
        #     "backbone.encoder.blocks.8.attention.dense2": self.backbone.encoder.blocks[8].attention.dense2,
        #     "backbone.encoder.blocks.8.attention.dense3": self.backbone.encoder.blocks[8].attention.dense3,
        #     "backbone.encoder.blocks.8.output": self.backbone.encoder.blocks[8].output,
        #     "backbone.encoder.blocks.8.output.mapping": self.backbone.encoder.blocks[8].output.mapping,
        #     "backbone.encoder.blocks.8.output.projection": self.backbone.encoder.blocks[8].output.projection,
        #     "backbone.encoder.blocks.8.output.dropout": self.backbone.encoder.blocks[8].output.dropout,
        #     "backbone.encoder.blocks.9": self.backbone.encoder.blocks[9],
        #     "backbone.encoder.blocks.9.layernorm1": self.backbone.encoder.blocks[9].layernorm1,
        #     "backbone.encoder.blocks.9.layernorm2": self.backbone.encoder.blocks[9].layernorm2,
        #     "backbone.encoder.blocks.9.attention": self.backbone.encoder.blocks[9].attention,
        #     "backbone.encoder.blocks.9.attention.projection": self.backbone.encoder.blocks[9].attention.projection,
        #     "backbone.encoder.blocks.9.attention.dropout": self.backbone.encoder.blocks[9].attention.dropout,
        #     "backbone.encoder.blocks.9.attention.prob_dropout": self.backbone.encoder.blocks[9].attention.prob_dropout,
        #     "backbone.encoder.blocks.9.attention.softmax": self.backbone.encoder.blocks[9].attention.softmax,
        #     "backbone.encoder.blocks.9.attention.dense1": self.backbone.encoder.blocks[9].attention.dense1,
        #     "backbone.encoder.blocks.9.attention.dense2": self.backbone.encoder.blocks[9].attention.dense2,
        #     "backbone.encoder.blocks.9.attention.dense3": self.backbone.encoder.blocks[9].attention.dense3,
        #     "backbone.encoder.blocks.9.output": self.backbone.encoder.blocks[9].output,
        #     "backbone.encoder.blocks.9.output.mapping": self.backbone.encoder.blocks[9].output.mapping,
        #     "backbone.encoder.blocks.9.output.projection": self.backbone.encoder.blocks[9].output.projection,
        #     "backbone.encoder.blocks.9.output.dropout": self.backbone.encoder.blocks[9].output.dropout,
        #     "backbone.encoder.blocks.10": self.backbone.encoder.blocks[10],
        #     "backbone.encoder.blocks.10.layernorm1": self.backbone.encoder.blocks[10].layernorm1,
        #     "backbone.encoder.blocks.10.layernorm2": self.backbone.encoder.blocks[10].layernorm2,
        #     "backbone.encoder.blocks.10.attention": self.backbone.encoder.blocks[10].attention,
        #     "backbone.encoder.blocks.10.attention.projection": self.backbone.encoder.blocks[10].attention.projection,
        #     "backbone.encoder.blocks.10.attention.dropout": self.backbone.encoder.blocks[10].attention.dropout,
        #     "backbone.encoder.blocks.10.attention.prob_dropout": self.backbone.encoder.blocks[
        #         10].attention.prob_dropout,
        #     "backbone.encoder.blocks.10.attention.softmax": self.backbone.encoder.blocks[10].attention.softmax,
        #     "backbone.encoder.blocks.10.attention.dense1": self.backbone.encoder.blocks[10].attention.dense1,
        #     "backbone.encoder.blocks.10.attention.dense2": self.backbone.encoder.blocks[10].attention.dense2,
        #     "backbone.encoder.blocks.10.attention.dense3": self.backbone.encoder.blocks[10].attention.dense3,
        #     "backbone.encoder.blocks.10.output": self.backbone.encoder.blocks[10].output,
        #     "backbone.encoder.blocks.10.output.mapping": self.backbone.encoder.blocks[10].output.mapping,
        #     "backbone.encoder.blocks.10.output.projection": self.backbone.encoder.blocks[10].output.projection,
        #     "backbone.encoder.blocks.10.output.dropout": self.backbone.encoder.blocks[10].output.dropout,
        #     "backbone.encoder.blocks.11": self.backbone.encoder.blocks[11],
        #     "backbone.encoder.blocks.11.layernorm1": self.backbone.encoder.blocks[11].layernorm1,
        #     "backbone.encoder.blocks.11.layernorm2": self.backbone.encoder.blocks[11].layernorm2,
        #     "backbone.encoder.blocks.11.attention": self.backbone.encoder.blocks[11].attention,
        #     "backbone.encoder.blocks.11.attention.projection": self.backbone.encoder.blocks[11].attention.projection,
        #     "backbone.encoder.blocks.11.attention.dropout": self.backbone.encoder.blocks[11].attention.dropout,
        #     "backbone.encoder.blocks.11.attention.prob_dropout": self.backbone.encoder.blocks[
        #         11].attention.prob_dropout,
        #     "backbone.encoder.blocks.11.attention.softmax": self.backbone.encoder.blocks[11].attention.softmax,
        #     "backbone.encoder.blocks.11.attention.dense1": self.backbone.encoder.blocks[11].attention.dense1,
        #     "backbone.encoder.blocks.11.attention.dense2": self.backbone.encoder.blocks[11].attention.dense2,
        #     "backbone.encoder.blocks.11.attention.dense3": self.backbone.encoder.blocks[11].attention.dense3,
        #     "backbone.encoder.blocks.11.output": self.backbone.encoder.blocks[11].output,
        #     "backbone.encoder.blocks.11.output.mapping": self.backbone.encoder.blocks[11].output.mapping,
        #     "backbone.encoder.blocks.11.output.projection": self.backbone.encoder.blocks[11].output.projection,
        #     "backbone.encoder.blocks.11.output.dropout": self.backbone.encoder.blocks[11].output.dropout,
        #     "backbone.encoder.blocks.12": self.backbone.encoder.blocks[12],
        #     "backbone.encoder.blocks.12.layernorm1": self.backbone.encoder.blocks[12].layernorm1,
        #     "backbone.encoder.blocks.12.layernorm2": self.backbone.encoder.blocks[12].layernorm2,
        #     "backbone.encoder.blocks.12.attention": self.backbone.encoder.blocks[12].attention,
        #     "backbone.encoder.blocks.12.attention.projection": self.backbone.encoder.blocks[12].attention.projection,
        #     "backbone.encoder.blocks.12.attention.dropout": self.backbone.encoder.blocks[12].attention.dropout,
        #     "backbone.encoder.blocks.12.attention.prob_dropout": self.backbone.encoder.blocks[
        #         12].attention.prob_dropout,
        #     "backbone.encoder.blocks.12.attention.softmax": self.backbone.encoder.blocks[12].attention.softmax,
        #     "backbone.encoder.blocks.12.attention.dense1": self.backbone.encoder.blocks[12].attention.dense1,
        #     "backbone.encoder.blocks.12.attention.dense2": self.backbone.encoder.blocks[12].attention.dense2,
        #     "backbone.encoder.blocks.12.attention.dense3": self.backbone.encoder.blocks[12].attention.dense3,
        #     "backbone.encoder.blocks.12.output": self.backbone.encoder.blocks[12].output,
        #     "backbone.encoder.blocks.12.output.mapping": self.backbone.encoder.blocks[12].output.mapping,
        #     "backbone.encoder.blocks.12.output.projection": self.backbone.encoder.blocks[12].output.projection,
        #     "backbone.encoder.blocks.12.output.dropout": self.backbone.encoder.blocks[12].output.dropout,
        #     "backbone.encoder.blocks.13": self.backbone.encoder.blocks[13],
        #     "backbone.encoder.blocks.13.layernorm1": self.backbone.encoder.blocks[13].layernorm1,
        #     "backbone.encoder.blocks.13.layernorm2": self.backbone.encoder.blocks[13].layernorm2,
        #     "backbone.encoder.blocks.13.attention": self.backbone.encoder.blocks[13].attention,
        #     "backbone.encoder.blocks.13.attention.projection": self.backbone.encoder.blocks[13].attention.projection,
        #     "backbone.encoder.blocks.13.attention.dropout": self.backbone.encoder.blocks[13].attention.dropout,
        #     "backbone.encoder.blocks.13.attention.prob_dropout": self.backbone.encoder.blocks[
        #         13].attention.prob_dropout,
        #     "backbone.encoder.blocks.13.attention.softmax": self.backbone.encoder.blocks[13].attention.softmax,
        #     "backbone.encoder.blocks.13.attention.dense1": self.backbone.encoder.blocks[13].attention.dense1,
        #     "backbone.encoder.blocks.13.attention.dense2": self.backbone.encoder.blocks[13].attention.dense2,
        #     "backbone.encoder.blocks.13.attention.dense3": self.backbone.encoder.blocks[13].attention.dense3,
        #     "backbone.encoder.blocks.13.output": self.backbone.encoder.blocks[13].output,
        #     "backbone.encoder.blocks.13.output.mapping": self.backbone.encoder.blocks[13].output.mapping,
        #     "backbone.encoder.blocks.13.output.projection": self.backbone.encoder.blocks[13].output.projection,
        #     "backbone.encoder.blocks.13.output.dropout": self.backbone.encoder.blocks[13].output.dropout,
        #     "backbone.encoder.blocks.14": self.backbone.encoder.blocks[14],
        #     "backbone.encoder.blocks.14.layernorm1": self.backbone.encoder.blocks[14].layernorm1,
        #     "backbone.encoder.blocks.14.layernorm2": self.backbone.encoder.blocks[14].layernorm2,
        #     "backbone.encoder.blocks.14.attention": self.backbone.encoder.blocks[14].attention,
        #     "backbone.encoder.blocks.14.attention.projection": self.backbone.encoder.blocks[14].attention.projection,
        #     "backbone.encoder.blocks.14.attention.dropout": self.backbone.encoder.blocks[14].attention.dropout,
        #     "backbone.encoder.blocks.14.attention.prob_dropout": self.backbone.encoder.blocks[
        #         14].attention.prob_dropout,
        #     "backbone.encoder.blocks.14.attention.softmax": self.backbone.encoder.blocks[14].attention.softmax,
        #     "backbone.encoder.blocks.14.attention.dense1": self.backbone.encoder.blocks[14].attention.dense1,
        #     "backbone.encoder.blocks.14.attention.dense2": self.backbone.encoder.blocks[14].attention.dense2,
        #     "backbone.encoder.blocks.14.attention.dense3": self.backbone.encoder.blocks[14].attention.dense3,
        #     "backbone.encoder.blocks.14.output": self.backbone.encoder.blocks[14].output,
        #     "backbone.encoder.blocks.14.output.mapping": self.backbone.encoder.blocks[14].output.mapping,
        #     "backbone.encoder.blocks.14.output.projection": self.backbone.encoder.blocks[14].output.projection,
        #     "backbone.encoder.blocks.14.output.dropout": self.backbone.encoder.blocks[14].output.dropout,
        #     "backbone.encoder.blocks.15": self.backbone.encoder.blocks[15],
        #     "backbone.encoder.blocks.15.layernorm1": self.backbone.encoder.blocks[15].layernorm1,
        #     "backbone.encoder.blocks.15.layernorm2": self.backbone.encoder.blocks[15].layernorm2,
        #     "backbone.encoder.blocks.15.attention": self.backbone.encoder.blocks[15].attention,
        #     "backbone.encoder.blocks.15.attention.projection": self.backbone.encoder.blocks[15].attention.projection,
        #     "backbone.encoder.blocks.15.attention.dropout": self.backbone.encoder.blocks[15].attention.dropout,
        #     "backbone.encoder.blocks.15.attention.prob_dropout": self.backbone.encoder.blocks[
        #         15].attention.prob_dropout,
        #     "backbone.encoder.blocks.15.attention.softmax": self.backbone.encoder.blocks[15].attention.softmax,
        #     "backbone.encoder.blocks.15.attention.dense1": self.backbone.encoder.blocks[15].attention.dense1,
        #     "backbone.encoder.blocks.15.attention.dense2": self.backbone.encoder.blocks[15].attention.dense2,
        #     "backbone.encoder.blocks.15.attention.dense3": self.backbone.encoder.blocks[15].attention.dense3,
        #     "backbone.encoder.blocks.15.output": self.backbone.encoder.blocks[15].output,
        #     "backbone.encoder.blocks.15.output.mapping": self.backbone.encoder.blocks[15].output.mapping,
        #     "backbone.encoder.blocks.15.output.projection": self.backbone.encoder.blocks[15].output.projection,
        #     "backbone.encoder.blocks.15.output.dropout": self.backbone.encoder.blocks[15].output.dropout,
        #     "backbone.encoder.blocks.16": self.backbone.encoder.blocks[16],
        #     "backbone.encoder.blocks.16.layernorm1": self.backbone.encoder.blocks[16].layernorm1,
        #     "backbone.encoder.blocks.16.layernorm2": self.backbone.encoder.blocks[16].layernorm2,
        #     "backbone.encoder.blocks.16.attention": self.backbone.encoder.blocks[16].attention,
        #     "backbone.encoder.blocks.16.attention.projection": self.backbone.encoder.blocks[16].attention.projection,
        #     "backbone.encoder.blocks.16.attention.dropout": self.backbone.encoder.blocks[16].attention.dropout,
        #     "backbone.encoder.blocks.16.attention.prob_dropout": self.backbone.encoder.blocks[
        #         16].attention.prob_dropout,
        #     "backbone.encoder.blocks.16.attention.softmax": self.backbone.encoder.blocks[16].attention.softmax,
        #     "backbone.encoder.blocks.16.attention.dense1": self.backbone.encoder.blocks[16].attention.dense1,
        #     "backbone.encoder.blocks.16.attention.dense2": self.backbone.encoder.blocks[16].attention.dense2,
        #     "backbone.encoder.blocks.16.attention.dense3": self.backbone.encoder.blocks[16].attention.dense3,
        #     "backbone.encoder.blocks.16.output": self.backbone.encoder.blocks[16].output,
        #     "backbone.encoder.blocks.16.output.mapping": self.backbone.encoder.blocks[16].output.mapping,
        #     "backbone.encoder.blocks.16.output.projection": self.backbone.encoder.blocks[16].output.projection,
        #     "backbone.encoder.blocks.16.output.dropout": self.backbone.encoder.blocks[16].output.dropout,
        #     "backbone.encoder.blocks.17": self.backbone.encoder.blocks[17],
        #     "backbone.encoder.blocks.17.layernorm1": self.backbone.encoder.blocks[17].layernorm1,
        #     "backbone.encoder.blocks.17.layernorm2": self.backbone.encoder.blocks[17].layernorm2,
        #     "backbone.encoder.blocks.17.attention": self.backbone.encoder.blocks[17].attention,
        #     "backbone.encoder.blocks.17.attention.projection": self.backbone.encoder.blocks[17].attention.projection,
        #     "backbone.encoder.blocks.17.attention.dropout": self.backbone.encoder.blocks[17].attention.dropout,
        #     "backbone.encoder.blocks.17.attention.prob_dropout": self.backbone.encoder.blocks[
        #         17].attention.prob_dropout,
        #     "backbone.encoder.blocks.17.attention.softmax": self.backbone.encoder.blocks[17].attention.softmax,
        #     "backbone.encoder.blocks.17.attention.dense1": self.backbone.encoder.blocks[17].attention.dense1,
        #     "backbone.encoder.blocks.17.attention.dense2": self.backbone.encoder.blocks[17].attention.dense2,
        #     "backbone.encoder.blocks.17.attention.dense3": self.backbone.encoder.blocks[17].attention.dense3,
        #     "backbone.encoder.blocks.17.output": self.backbone.encoder.blocks[17].output,
        #     "backbone.encoder.blocks.17.output.mapping": self.backbone.encoder.blocks[17].output.mapping,
        #     "backbone.encoder.blocks.17.output.projection": self.backbone.encoder.blocks[17].output.projection,
        #     "backbone.encoder.blocks.17.output.dropout": self.backbone.encoder.blocks[17].output.dropout,
        #     "backbone.encoder.blocks.18": self.backbone.encoder.blocks[18],
        #     "backbone.encoder.blocks.18.layernorm1": self.backbone.encoder.blocks[18].layernorm1,
        #     "backbone.encoder.blocks.18.layernorm2": self.backbone.encoder.blocks[18].layernorm2,
        #     "backbone.encoder.blocks.18.attention": self.backbone.encoder.blocks[18].attention,
        #     "backbone.encoder.blocks.18.attention.projection": self.backbone.encoder.blocks[18].attention.projection,
        #     "backbone.encoder.blocks.18.attention.dropout": self.backbone.encoder.blocks[18].attention.dropout,
        #     "backbone.encoder.blocks.18.attention.prob_dropout": self.backbone.encoder.blocks[
        #         18].attention.prob_dropout,
        #     "backbone.encoder.blocks.18.attention.softmax": self.backbone.encoder.blocks[18].attention.softmax,
        #     "backbone.encoder.blocks.18.attention.dense1": self.backbone.encoder.blocks[18].attention.dense1,
        #     "backbone.encoder.blocks.18.attention.dense2": self.backbone.encoder.blocks[18].attention.dense2,
        #     "backbone.encoder.blocks.18.attention.dense3": self.backbone.encoder.blocks[18].attention.dense3,
        #     "backbone.encoder.blocks.18.output": self.backbone.encoder.blocks[18].output,
        #     "backbone.encoder.blocks.18.output.mapping": self.backbone.encoder.blocks[18].output.mapping,
        #     "backbone.encoder.blocks.18.output.projection": self.backbone.encoder.blocks[18].output.projection,
        #     "backbone.encoder.blocks.18.output.dropout": self.backbone.encoder.blocks[18].output.dropout,
        #     "backbone.encoder.blocks.19": self.backbone.encoder.blocks[19],
        #     "backbone.encoder.blocks.19.layernorm1": self.backbone.encoder.blocks[19].layernorm1,
        #     "backbone.encoder.blocks.19.layernorm2": self.backbone.encoder.blocks[19].layernorm2,
        #     "backbone.encoder.blocks.19.attention": self.backbone.encoder.blocks[19].attention,
        #     "backbone.encoder.blocks.19.attention.projection": self.backbone.encoder.blocks[19].attention.projection,
        #     "backbone.encoder.blocks.19.attention.dropout": self.backbone.encoder.blocks[19].attention.dropout,
        #     "backbone.encoder.blocks.19.attention.prob_dropout": self.backbone.encoder.blocks[
        #         19].attention.prob_dropout,
        #     "backbone.encoder.blocks.19.attention.softmax": self.backbone.encoder.blocks[19].attention.softmax,
        #     "backbone.encoder.blocks.19.attention.dense1": self.backbone.encoder.blocks[19].attention.dense1,
        #     "backbone.encoder.blocks.19.attention.dense2": self.backbone.encoder.blocks[19].attention.dense2,
        #     "backbone.encoder.blocks.19.attention.dense3": self.backbone.encoder.blocks[19].attention.dense3,
        #     "backbone.encoder.blocks.19.output": self.backbone.encoder.blocks[19].output,
        #     "backbone.encoder.blocks.19.output.mapping": self.backbone.encoder.blocks[19].output.mapping,
        #     "backbone.encoder.blocks.19.output.projection": self.backbone.encoder.blocks[19].output.projection,
        #     "backbone.encoder.blocks.19.output.dropout": self.backbone.encoder.blocks[19].output.dropout,
        #     "backbone.encoder.blocks.20": self.backbone.encoder.blocks[20],
        #     "backbone.encoder.blocks.20.layernorm1": self.backbone.encoder.blocks[20].layernorm1,
        #     "backbone.encoder.blocks.20.layernorm2": self.backbone.encoder.blocks[20].layernorm2,
        #     "backbone.encoder.blocks.20.attention": self.backbone.encoder.blocks[20].attention,
        #     "backbone.encoder.blocks.20.attention.projection": self.backbone.encoder.blocks[20].attention.projection,
        #     "backbone.encoder.blocks.20.attention.dropout": self.backbone.encoder.blocks[20].attention.dropout,
        #     "backbone.encoder.blocks.20.attention.prob_dropout": self.backbone.encoder.blocks[
        #         20].attention.prob_dropout,
        #     "backbone.encoder.blocks.20.attention.softmax": self.backbone.encoder.blocks[20].attention.softmax,
        #     "backbone.encoder.blocks.20.attention.dense1": self.backbone.encoder.blocks[20].attention.dense1,
        #     "backbone.encoder.blocks.20.attention.dense2": self.backbone.encoder.blocks[20].attention.dense2,
        #     "backbone.encoder.blocks.20.attention.dense3": self.backbone.encoder.blocks[20].attention.dense3,
        #     "backbone.encoder.blocks.20.output": self.backbone.encoder.blocks[20].output,
        #     "backbone.encoder.blocks.20.output.mapping": self.backbone.encoder.blocks[20].output.mapping,
        #     "backbone.encoder.blocks.20.output.projection": self.backbone.encoder.blocks[20].output.projection,
        #     "backbone.encoder.blocks.20.output.dropout": self.backbone.encoder.blocks[20].output.dropout,
        #     "backbone.encoder.blocks.21": self.backbone.encoder.blocks[21],
        #     "backbone.encoder.blocks.21.layernorm1": self.backbone.encoder.blocks[21].layernorm1,
        #     "backbone.encoder.blocks.21.layernorm2": self.backbone.encoder.blocks[21].layernorm2,
        #     "backbone.encoder.blocks.21.attention": self.backbone.encoder.blocks[21].attention,
        #     "backbone.encoder.blocks.21.attention.projection": self.backbone.encoder.blocks[21].attention.projection,
        #     "backbone.encoder.blocks.21.attention.dropout": self.backbone.encoder.blocks[21].attention.dropout,
        #     "backbone.encoder.blocks.21.attention.prob_dropout": self.backbone.encoder.blocks[
        #         21].attention.prob_dropout,
        #     "backbone.encoder.blocks.21.attention.softmax": self.backbone.encoder.blocks[21].attention.softmax,
        #     "backbone.encoder.blocks.21.attention.softmax_3d": self.backbone.encoder.blocks[21].attention.softmax_3d,
        #     "backbone.encoder.blocks.21.attention.dense1": self.backbone.encoder.blocks[21].attention.dense1,
        #     "backbone.encoder.blocks.21.attention.dense2": self.backbone.encoder.blocks[21].attention.dense2,
        #     "backbone.encoder.blocks.21.attention.dense3": self.backbone.encoder.blocks[21].attention.dense3,
        #     "backbone.encoder.blocks.21.output": self.backbone.encoder.blocks[21].output,
        #     "backbone.encoder.blocks.21.output.mapping": self.backbone.encoder.blocks[21].output.mapping,
        #     "backbone.encoder.blocks.21.output.projection": self.backbone.encoder.blocks[21].output.projection,
        #     "backbone.encoder.blocks.21.output.dropout": self.backbone.encoder.blocks[21].output.dropout,
        #     "backbone.encoder.blocks.22": self.backbone.encoder.blocks[22],
        #     "backbone.encoder.blocks.22.layernorm1": self.backbone.encoder.blocks[22].layernorm1,
        #     "backbone.encoder.blocks.22.layernorm2": self.backbone.encoder.blocks[22].layernorm2,
        #     "backbone.encoder.blocks.22.attention": self.backbone.encoder.blocks[22].attention,
        #     "backbone.encoder.blocks.22.attention.projection": self.backbone.encoder.blocks[22].attention.projection,
        #     "backbone.encoder.blocks.22.attention.dropout": self.backbone.encoder.blocks[22].attention.dropout,
        #     "backbone.encoder.blocks.22.attention.prob_dropout": self.backbone.encoder.blocks[
        #         22].attention.prob_dropout,
        #     "backbone.encoder.blocks.22.attention.softmax": self.backbone.encoder.blocks[22].attention.softmax,
        #     "backbone.encoder.blocks.22.attention.dense1": self.backbone.encoder.blocks[22].attention.dense1,
        #     "backbone.encoder.blocks.22.attention.dense2": self.backbone.encoder.blocks[22].attention.dense2,
        #     "backbone.encoder.blocks.22.attention.dense3": self.backbone.encoder.blocks[22].attention.dense3,
        #     "backbone.encoder.blocks.22.output": self.backbone.encoder.blocks[22].output,
        #     "backbone.encoder.blocks.22.output.mapping": self.backbone.encoder.blocks[22].output.mapping,
        #     "backbone.encoder.blocks.22.output.projection": self.backbone.encoder.blocks[22].output.projection,
        #     "backbone.encoder.blocks.22.output.dropout": self.backbone.encoder.blocks[22].output.dropout,
        #     "backbone.encoder.blocks.23": self.backbone.encoder.blocks[23],
        #     "backbone.encoder.blocks.23.layernorm1": self.backbone.encoder.blocks[23].layernorm1,
        #     "backbone.encoder.blocks.23.layernorm2": self.backbone.encoder.blocks[23].layernorm2,
        #     "backbone.encoder.blocks.23.attention": self.backbone.encoder.blocks[23].attention,
        #     "backbone.encoder.blocks.23.attention.projection": self.backbone.encoder.blocks[23].attention.projection,
        #     "backbone.encoder.blocks.23.attention.dropout": self.backbone.encoder.blocks[23].attention.dropout,
        #     "backbone.encoder.blocks.23.attention.prob_dropout": self.backbone.encoder.blocks[
        #         23].attention.prob_dropout,
        #     "backbone.encoder.blocks.23.attention.softmax": self.backbone.encoder.blocks[23].attention.softmax,
        #     "backbone.encoder.blocks.23.attention.dense1": self.backbone.encoder.blocks[23].attention.dense1,
        #     "backbone.encoder.blocks.23.attention.dense2": self.backbone.encoder.blocks[23].attention.dense2,
        #     "backbone.encoder.blocks.23.attention.dense3": self.backbone.encoder.blocks[23].attention.dense3,
        #     "backbone.encoder.blocks.23.output": self.backbone.encoder.blocks[23].output,
        #     "backbone.encoder.blocks.23.output.mapping": self.backbone.encoder.blocks[23].output.mapping,
        #     "backbone.encoder.blocks.23.output.projection": self.backbone.encoder.blocks[23].output.projection,
        #     "backbone.encoder.blocks.23.output.dropout": self.backbone.encoder.blocks[23].output.dropout,
            "backbone.layernorm": self.backbone.layernorm,
            "head": self.head,
        }

    def construct(self, input_ids):
        tokens = input_ids[:, :-1]
        input_mask = F.cast(F.not_equal(tokens, self.eos_token), mstype.float32)
        output_states, _, embedding_table = self.backbone(tokens, input_mask)
        # print("output_states", output_states.shape)
        # print("embedding_table", embedding_table.shape)
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
            self.backbone.encoder.blocks[8].attention.dense2 = new_layer
            self.layer_names["backbone.encoder.blocks.8.attention.dense2"]=new_layer

        elif 'backbone.encoder.blocks.8.attention.dense3' == layer_name:
            self.backbone.encoder.blocks[8].attention.dense3 = new_layer
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

# if __name__ == '__main__':
#     from gpt_torch import device
#     mindspore.set_context(pynative_synchronize=True)
#     model = GPT()
#     # mindspore.load_checkpoint('./convert_ms.ckpt', model)
#     model.set_train(False)
#     model_torch = GPTPyTorch(config).to(device)
#     # model_torch.load_state_dict(torch.load('./torch_net.path'))
#     model_torch.eval()
#     # loss = CrossEntropyLoss()
#     # model = GPTWithLoss(model, loss)
#     np_data = [np.ones((1, 1025))]
#     dtypes = [mindspore.int32]
#     # input_data = mindsporeinfoplus.np_2_tensor(np_data, dtypes)
#     # res, global_layer_info = mindsporeinfoplus.summary_plus(
#     #     model=model,
#     #     input_data=input_data,
#     #     dtypes=dtypes,
#     #     col_names=['input_size', 'output_size', 'name'],
#     #     verbose=2,
#     #     depth=10)
#     # input_datas = mindsporeinfoplus.get_input_datas(global_layer_info)
#     anp = (mindspore.numpy.ones((1, 1025)).astype(mindspore.int64),)
#     diff_finder = ts.migrator.NetDifferenceFinder(pt_net=model_torch, ms_net=model, fix_seed=0, auto_conv_ckpt=2)  #
#     ts.migrator.get_weight_map(pt_net=model_torch, weight_map_save_path='./torch_net_map.json', print_map=False)
#
#     torch.save(model_torch.state_dict(), './torch_net.path')
#
#     ts.migrator.convert_weight(weight_map_path='./torch_net_map.json', pt_file_path='./torch_net.path',
#                                ms_file_save_path='./convert_ms.ckpt', print_conv_info=False)
#
#     param_dict = mindspore.load_checkpoint('./convert_ms.ckpt')
#     mindspore.load_param_into_net(model, param_dict)
#     res, _, _ = compare_models_new(model, model_torch, np_data, dtypes, [torch.int64])
#     print("res", res)
#     # a = mindspore.Tensor(np_data[0], dtype=mindspore.int32)
#     # tokens = a[:, :-1]
#     # input_mask = F.cast(F.not_equal(tokens, 50256), mstype.float32)
#     # res = model(tokens, input_mask)
#     # res = model(a)
#     # print(res.shape)
#     # print(res[0].shape)
#     # print(res[1][0].shape)
#     # print(res[2].shape)
