import paddle 
import numpy as np 
from paddle import Tensor, nn

from custom_setup_ops import neox_rope

dtype = paddle.float32 
batchsize = 1
seq_len = 16
numhead = 4
head_size = 8

np.random.seed(0)

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

qkv = paddle.cast(paddle.to_tensor(np.random.randn(batchsize, seq_len, numhead, 3 * head_size)), dtype)
rotary_embedding = paddle.cast(get_rotary_embedding(batchsize, 2048, 10000, head_size, seq_len, offset=0), paddle.float32)
# print(rotary_embedding.shape)
print("Get rotary embedding: ", rotary_embedding)
qkv_out = neox_rope(qkv, rotary_embedding)
print(qkv_out.shape)
q_out = qkv_out[:, :, :, :head_size]
k_out = qkv_out[:, :, :, head_size:head_size*2]

baseline_q_out, baseline_k_out = baseline(qkv[:, :, :, :head_size], 
                                          qkv[:, :, :, head_size:head_size*2], batchsize, 2048, 10000, head_size, seq_len, offset=0)

print("Q is equal?: ", np.allclose(q_out.numpy(), baseline_q_out.numpy(), atol=1e-3, rtol=1e-3))
print("K is equal?: ", np.allclose(k_out.numpy(), baseline_k_out.numpy(), atol=1e-3, rtol=1e-3))
