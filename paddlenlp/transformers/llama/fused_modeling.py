# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Paddle Llama model"""
from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple, Type

import numpy as np
import paddle
import paddle.distributed.fleet.meta_parallel as mpu
import paddle.nn.functional as F
from paddle import Tensor, nn
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute
from paddle.incubate.nn import FusedMultiTransformer
from paddle.utils import try_import

from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)

try:
    from paddle.nn.functional.flash_attention import flash_attention
except:
    flash_attention = None

import warnings

from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from paddlenlp.transformers.model_utils import PretrainedModel, register_base_model

from .configuration import LlamaConfig

LLAMA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "__internal_testing__/tiny-random-llama",
    "facebook/llama-7b",
    "facebook/llama-13b",
]

__all__ = ["LlamaModel", "LlamaPretrainedModel", "LlamaForCausalLM", "LlamaPretrainingCriterion", "FusedLlamaModel"]



class FusedLlamaRMSNorm(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = config.rms_norm_eps
        self.config = config

    def forward(self, hidden_states, residual=None, bias=None):
        return paddle._C_ops.norm_helper(
            hidden_states,
            residual,
            bias,
            self.weight,
            None,
            self.variance_epsilon,
            1.0, # residual_alpha(not used.)
            "rmsnorm",
            begin_norm_axis=2,
        )[0:2]
