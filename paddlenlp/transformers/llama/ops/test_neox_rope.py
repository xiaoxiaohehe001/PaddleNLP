import paddle 
import numpy as np 
from paddle import Tensor, nn

from custom_setup_ops import neox_rope, get_padding_offset

np.random.seed(0)

dtype = paddle.float32 
numhead = 4
seq_length =[1, 3, 10]
# seq_length =[1, 2]

batchsize = len(seq_length)
head_size = 16

max_seq_len = np.max(seq_length)

paddle_seq_length = paddle.cast(paddle.to_tensor(seq_length), dtype=paddle.int32)


class LlamaRotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        dtype = paddle.get_default_dtype()
        inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, dim, 2), dtype="float32") / dim))
        self.register_buffer("inv_freq", inv_freq.cast(dtype))

        # higher acc using float32
        t = paddle.arange(max_position_embeddings, dtype="float32")
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq.cast("float32"))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [bs, seqlen, nhead, head_dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, x, seq_len=None):
        return (
            self.cos_cached[:, :seq_len, :, ...],
            self.sin_cached[:, :seq_len, :, ...],
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[:, offset : q.shape[1] + offset, :, :]
    sin = sin[:, offset : q.shape[1] + offset, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def baseline(q, k, bsz, max_position_embeddings, base, head_dim, seq_length, offset): 
    rotary_emb = LlamaRotaryEmbedding(head_dim, max_position_embeddings)
    cos, sin = rotary_emb(None, seq_len=seq_length)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, offset=offset)
    return q_out, k_out 


def pad_batch_data(insts, max_seq_len, pad_id=0, return_seq_len=False, pad_style="right"):
    """Pad sequences to the max sequence length in batch. """
    max_len = max_seq_len
    if pad_style == "left":
        inst_data = np.array([[pad_id] * (max_len - len(inst)) + list(inst) for inst in insts])
    else:
        inst_data = np.array([list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])
    if return_seq_len:
        seq_len = np.array([len(inst) for inst in insts])
        return inst_data.astype("int64").reshape([-1, max_len]), seq_len
    else:
        return inst_data.astype("int64").reshape([-1, max_len])


def remove_padding(input_ids, seq_lens_this_time):
    max_len = paddle.max(seq_lens_this_time)
    cum_offsets_now = paddle.cumsum(max_len - seq_lens_this_time)
    token_num = paddle.sum(seq_lens_this_time)
    ids_remove_padding, cum_offsets, padding_offset = get_padding_offset(input_ids, cum_offsets_now, token_num, seq_lens_this_time)
    return ids_remove_padding, padding_offset, cum_offsets


def set_qkv_padding(qkv, seq_length, max_seq_len):
    for idx, seq_len in enumerate(seq_length): 
        print(seq_len, max_seq_len)
        qkv[idx, seq_len:max_seq_len, :, :] = 0
    return qkv 

def get_flatten_qkv(qkv, seq_length, max_seq_len, numhead, headsize):
    token_num = sum(seq_length)
    flatten_qkv = paddle.zeros((token_num, numhead, 3 * headsize), qkv.dtype)
    token_start = 0

    for idx, seq_len in enumerate(seq_length): 
        flatten_qkv[token_start:token_start + seq_len, :, :] = qkv[idx, :seq_len, :, :]
        token_start += seq_len 
    return flatten_qkv 

def get_flatten_remove_padding_rope(q, seq_length, max_seq_len, numhead, headsize):
    token_num = sum(seq_length)
    flatten_qkv = paddle.zeros((token_num, numhead, headsize), qkv.dtype)
    token_start = 0

    for idx, seq_len in enumerate(seq_length): 
        flatten_qkv[token_start:token_start + seq_len, :, :] = q[idx, :seq_len, :, :]
        token_start += seq_len 
    return flatten_qkv

def get_rotary_embedding(bsz, max_position_embeddings, base, head_dim, seq_length, offset): 
    inv_freq = 1.0 / (base ** (paddle.cast(paddle.arange(0, head_dim, 2), "float32") / head_dim))
    t = paddle.arange(max_position_embeddings, dtype=inv_freq.dtype)

    # shape: [S, D/2]
    freqs = paddle.einsum("i,j->ij", t, inv_freq)
    # shape: [S, D]
    emb = paddle.concat([freqs, freqs], axis=-1)

    # shape: [1, S, D]
    emb = paddle.unsqueeze(emb, 0)
    # shape: [1, S, 1, D]
    emb = paddle.unsqueeze(emb, 2)
    # shape: [B, S, 1, D]
    emb = paddle.repeat_interleave(emb, bsz, axis=0)
    emb = paddle.transpose(emb, perm=[0, 2, 1, 3])
    cos_emb = paddle.cos(emb)
    sin_emb = paddle.sin(emb)

    stacked_rotary_emb = paddle.concat([cos_emb, sin_emb], axis=0)
    return stacked_rotary_emb[:, :, offset: seq_length+offset, :]

batch_data = []
for seq in seq_length: 
    batch_data.append(np.ones((seq)))

padded_data = pad_batch_data(batch_data, max_seq_len)
paddle_input_id = paddle.cast(paddle.to_tensor(padded_data), dtype=paddle.int64)

ids_remove_padding, padding_offset, cum_offsets = remove_padding(paddle_input_id, paddle_seq_length)
print("Padding offset is: ", padding_offset)
print("Cum offset is: ", cum_offsets)

rotary_embedding = paddle.cast(get_rotary_embedding(batchsize, 2048, 10000, head_size, max_seq_len, offset=0), paddle.float32)
print("rotary emb shape is: ", rotary_embedding.shape)

np_qkv = np.random.randn(batchsize, max_seq_len, numhead, 3 * head_size)
origin_qkv = paddle.cast(paddle.to_tensor(np_qkv), dtype)
qkv = paddle.cast(paddle.to_tensor(np_qkv), dtype)
padded_qkv = set_qkv_padding(qkv, seq_length, max_seq_len)


flatten_qkv = get_flatten_qkv(qkv, seq_length, max_seq_len, numhead, head_size)

print("Original Q is: ", flatten_qkv[:, :, :head_size])
print("Original K is: ", flatten_qkv[:, :, head_size: 2 * head_size])

baseline_q_out, baseline_k_out = baseline(origin_qkv[:, :, :, :head_size], 
                                          origin_qkv[:, :, :, head_size:head_size*2], batchsize, 2048, 10000, head_size, max_seq_len, offset=0)

baseline_q_out_flatten = get_flatten_remove_padding_rope(baseline_q_out, seq_length, max_seq_len, numhead, head_size)
baseline_k_out_flatten = get_flatten_remove_padding_rope(baseline_k_out, seq_length, max_seq_len, numhead, head_size)

print("Baseline q out flatten: ", baseline_q_out_flatten)
print("Baseline k out flatten: ", baseline_k_out_flatten)

neox_rope(flatten_qkv, rotary_embedding, padding_offset, paddle_seq_length)
print("After Custom neox rope Q: ", flatten_qkv[:, :, :head_size])
print("After Custom neox rope K: ", flatten_qkv[:, :, head_size:2*head_size])


print("Is Q equal?: ", np.allclose(baseline_q_out_flatten.numpy(), flatten_qkv[:, :, :head_size].numpy(), atol=1e-3, rtol=1e-3))
print("Is K equal?: ", np.allclose(baseline_k_out_flatten.numpy(), flatten_qkv[:, :, head_size:2*head_size].numpy(), atol=1e-3, rtol=1e-3))
