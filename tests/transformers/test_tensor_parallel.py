# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import tempfile

import paddle
from paddle.distributed import fleet

tp_size = paddle.distributed.get_world_size()
tp_rank = 0
if tp_size > 1:
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tp_size,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    fleet.init(is_collective=True, strategy=strategy)
    hcg = fleet.get_hybrid_communicate_group()
    tp_rank = hcg.get_model_parallel_rank()
    mp_group = hcg.get_model_parallel_group()


def prepare_config(config):
    config.hidden_size = 512
    config.num_layers = 2
    config.num_attention_heads = 16
    config.intermediate_size = config.hidden_size * 3
    config.tensor_parallel_degree = tp_size
    config.tensor_parallel_rank = tp_rank
    return config


def test_load(model_class, tempdir):
    paddle.distributed.barrier()
    if model_class is not None:
        model_class.from_pretrained(tempdir)
        paddle.distributed.barrier()
        if paddle.distributed.get_rank() == 0:
            files = glob.glob(tempdir + "/*")
            for f in files:
                os.remove(f)


def test_merge(model, model_class=None):
    object_list = []
    with tempfile.TemporaryDirectory() as tempdir:
        paddle.distributed.all_gather_object(object_list, tempdir, group=mp_group)
        tempdir = object_list[0]
        # test merge one
        model.save_pretrained(save_dir=tempdir, merge_tensor_parallel=True)
        test_load(model_class, tempdir)
        # test merge shard
        model.save_pretrained(tempdir, merge_tensor_parallel=True, variant="tp00", max_shard_size="5MB")
        test_load(model_class, tempdir)
        # test save tp
        model.save_pretrained(tempdir, max_shard_size="5MB")
        test_load(model_class, tempdir)
        # test save shard safe
        model.save_pretrained(tempdir, max_shard_size="5MB", safe_serialization=True)
        test_load(model_class, tempdir)
        paddle.distributed.barrier()
        # print( "\n".join(sorted(os.listdir(tempdir)) ))


def test_llama():
    from paddlenlp.transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig()
    config = prepare_config(config)
    model = LlamaForCausalLM._from_config(config)
    test_merge(model, LlamaForCausalLM)


def test_chatglm():
    from paddlenlp.transformers import ChatGLMConfig, ChatGLMForConditionalGeneration

    config = ChatGLMConfig()
    config = prepare_config(config)
    model = ChatGLMForConditionalGeneration._from_config(config)
    # print("\n".join(list(model.state_dict().keys())) )
    test_merge(model, ChatGLMForConditionalGeneration)


def test_bloom():
    from paddlenlp.transformers import BloomConfig, BloomForCausalLM

    config = BloomConfig()
    config = prepare_config(config)
    model = BloomForCausalLM._from_config(config)
    # print("\n".join(list(model.state_dict().keys())) )
    test_merge(model, BloomForCausalLM)


# test_llama()
test_chatglm()
test_bloom()

# from transformers import BloomForCausalLM, BloomConfig
# config = BloomConfig()
# config = prepare_config(config)
# model = BloomForCausalLM._from_config(config)
# print("\n".join(list(model.state_dict().keys())))
