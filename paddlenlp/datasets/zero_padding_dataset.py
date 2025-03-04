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

import numpy as np
from paddle.io import Dataset, IterableDataset
from scipy.linalg import block_diag


class ZeroPadding:
    required_output_keys = ["input_ids", "labels", "attention_mask"]
    # Only supported the following keys for ZeroPadding. Keys outside of the set will be ignored.
    supported_input_keys = [
        "input_ids",
        "labels",
        "attention_mask",
        "position_ids",
        "chosen_labels",
        "rejected_labels",
        "response_indexs",
        "attn_mask_start_row_indices",
    ]

    @classmethod
    def _pad_batch_records(cls, batch_records):
        # Only consider supported input keys
        input_keys = [key for key in batch_records[0].keys() if key in cls.supported_input_keys]
        if "attn_mask_start_row_indices" not in input_keys and "attention_mask" not in input_keys:
            input_keys.append("attention_mask")
        batched_features = {key: [] for key in input_keys}
        sequence_sum = 0
        for record in batch_records:
            batched_features["input_ids"].extend(record["input_ids"])
            if "labels" in record:
                batched_features["labels"].extend(record["labels"])
            elif "rejected_labels" in input_keys and "chosen_labels" in input_keys:
                batched_features["rejected_labels"].extend(record["rejected_labels"])
                batched_features["chosen_labels"].extend(record["chosen_labels"])
                response_indexs = [
                    record["response_indexs"][0] + sequence_sum,  # chosen_response_start_index
                    record["response_indexs"][1] + sequence_sum,  # rejeted_response_start_index
                    record["response_indexs"][2] + sequence_sum,  # rejeted_response_end_index + 1
                ]
                batched_features["response_indexs"].append(response_indexs)
            else:
                raise ValueError("labels is required for ZeroPadding Dataset")

            seq_length = len(record["input_ids"])
            # If attention_mask is not given, assume it's causal mask
            if "attn_mask_start_row_indices" in record:
                attn_mask_start_row_indices = [i + sequence_sum for i in record["attn_mask_start_row_indices"]]
                batched_features["attn_mask_start_row_indices"].extend(attn_mask_start_row_indices)
            else:
                attention_mask = record.get("attention_mask", np.tril(np.ones([seq_length, seq_length], dtype=bool)))
                batched_features["attention_mask"].append(attention_mask)
            # NOTE: position_ids is optional and not required by every model
            # We append instead of extend here to accomodate 2D position ids
            if "position_ids" in record:
                batched_features["position_ids"].append(record["position_ids"])
            sequence_sum += seq_length

        if "attention_mask" in batched_features:
            block_attention_mask = block_diag(*batched_features["attention_mask"])
            # convert to 3-D [batch_size(1), seq_length, seq_length]
            batched_features["attention_mask"] = np.expand_dims(block_attention_mask, axis=0)
        if "position_ids" in batched_features:
            # Accomodate both 1D and 2D position ids
            batched_features["position_ids"] = np.concatenate(batched_features["position_ids"], axis=-1).tolist()
        return batched_features


class ZeroPaddingMapDataset(ZeroPadding, Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.new_data = self._create_zero_padding_data(data)

    def _create_zero_padding_data(self, data):
        batch_records, max_len = [], 0
        cur_len_so_far = 0

        total_data = []
        for i in range(len(data)):
            record = data[i]
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_length
            if to_append:
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])
            else:
                # exceed max length
                padded_list = self._pad_batch_records(batch_records)
                total_data.append(padded_list)
                # reset
                batch_records, max_len = [], 0
                cur_len_so_far = 0
                # append current data
                batch_records.append(record)
                cur_len_so_far += len(record["input_ids"])

        # remaining data
        if batch_records:
            padded_list = self._pad_batch_records(batch_records)
            total_data.append(padded_list)
        return total_data

    def __getitem__(self, idx):
        return self.new_data[idx]

    def __len__(self):
        return len(self.new_data)


class ZeroPaddingIterableDataset(ZeroPadding, IterableDataset):
    def __init__(self, data, tokenizer, max_length):

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.zero_padding_global_step = 0

    def __iter__(self):
        batch_records, max_len = [], 0
        cur_len_so_far = 0
        for record in self.data:
            max_len = max(max_len, len(record["input_ids"]))
            to_append = (cur_len_so_far + len(record["input_ids"])) <= self.max_length
            if to_append:
                batch_records.append(record)
                self.zero_padding_global_step += 1
                cur_len_so_far += len(record["input_ids"])
            else:
                # exceed max length
                padded_list = self._pad_batch_records(batch_records)
                yield padded_list
                # reset
                batch_records, max_len = [], 0
                cur_len_so_far = 0
                # append current data
                batch_records.append(record)
                self.intokens_global_step += 1
                cur_len_so_far += len(record["input_ids"])
        if batch_records:
            padded_list = self._pad_batch_records(batch_records)
            yield padded_list
