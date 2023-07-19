import paddle 
import numpy as np 
from paddle import Tensor, nn

from custom_setup_ops import write_cache_kv

dtype = paddle.bfloat16
batch_size = 1
num_head = 2
seq_len = 1
head_size = 16
max_seq_len = 4


input_kv = paddle.cast(paddle.to_tensor(np.random.randn(2, batch_size, num_head, seq_len, head_size)), dtype)
cache_kv = paddle.cast(paddle.to_tensor(np.zeros([2, batch_size, num_head, max_seq_len, head_size])), dtype)
seq_data = paddle.cast(paddle.to_tensor(np.array([1])), "int32")

cache_kv_out = write_cache_kv(input_kv, cache_kv, seq_data)

input_k = input_kv[0].reshape([batch_size, num_head, seq_len, head_size//8, 8])
input_v = input_kv[1].reshape([batch_size, num_head, seq_len, head_size//8, 8])
zeros_tensor = paddle.cast(paddle.to_tensor(np.zeros([batch_size, num_head, 3, head_size//8, 8])), dtype)

k_out = cache_kv[0]
baseline_k_out = paddle.concat([input_k, zeros_tensor], axis=2).transpose([0, 1, 3, 2, 4])

v_out = cache_kv[1]
baseline_v_out = paddle.concat([input_v, zeros_tensor], axis=2)

print("k: ", np.allclose(k_out.reshape([1,-1]).numpy(), baseline_k_out.reshape([1,-1]).numpy(), atol=1e-3, rtol=1e-3))
print("v: ", np.allclose(v_out.reshape([1,-1]).numpy(), baseline_v_out.reshape([1,-1]).numpy(), atol=1e-3, rtol=1e-3))
