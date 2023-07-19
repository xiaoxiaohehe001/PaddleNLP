#include "paddle/extension.h"

template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
public:
  typedef __nv_bfloat16 DataType;
  typedef paddle::bfloat16 data_t;
};

template <typename T>
__global__ void NeoXRotaryKernel(const T *input,
                                 const float *cos_emb,
                                 const float *sin_emb,
                                 const int *sequence_lengths,
                                 T *output,
                                 const int rotary_emb_dims,
                                 const int batch_size,
                                 const int seq_len,
                                 const int num_head,
                                 const int dim_head,
                                 const int dim_head_mul3,
                                 const int last_dim) {
  int bi = blockIdx.x; // batch_idx
  int si = blockIdx.y; // seq_idx
  int hi = blockIdx.z; // head_idx

  if (sequence_lengths && si >= sequence_lengths[bi] * rotary_emb_dims) return;
  int half_lastdim = last_dim / 2;
  for(int qk_idx = 0; qk_idx < 2; qk_idx++){
    for (int ti = threadIdx.x; ti < half_lastdim; ti += blockDim.x) {
        int base_idx = bi * seq_len * num_head * dim_head_mul3 +
                       si * num_head * dim_head_mul3 + 
                       hi * dim_head_mul3 + qk_idx * dim_head; 

        int left_idx = base_idx + ti;
        const int right_idx = base_idx + ti + half_lastdim;
        int emb_idx_left = bi * seq_len * last_dim + si * last_dim + ti;
        int emb_idx_right =
            bi * seq_len * last_dim + si * last_dim + ti + half_lastdim;
        float input_left = static_cast<float>(input[left_idx]);
        float input_right = static_cast<float>(input[right_idx]);

        float cos_tmp_left = cos_emb[emb_idx_left];
        float sin_tmp_left = sin_emb[emb_idx_left];
        float cos_tmp_right = cos_emb[emb_idx_right];
        float sin_tmp_right = sin_emb[emb_idx_right];

        T res1 =
            static_cast<T>(input_left * cos_tmp_left - input_right * sin_tmp_left);
        T res2 = static_cast<T>(input_right * cos_tmp_right +
                                input_left * sin_tmp_right);
        output[left_idx] = res1;
        output[right_idx] = res2;
    }
  }
}


template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchRotaryQK(const paddle::Tensor& qkv, 
                                           const paddle::Tensor& rotary_emb) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int rotary_emb_dims = 1; // Since LLAMA Neox RotaryEmbedding no need rotary_emb_dims. Author(Zhengzekang)
    const int64_t batch_size = qkv.shape()[0]; 
    const int64_t seq_len = qkv.shape()[1]; 
    const int64_t num_head = qkv.shape()[2]; 
    const int64_t dim_head_mul3 = qkv.shape()[3]; 
    const int64_t dim_head = dim_head_mul3 / 3; 

    auto qkv_out = paddle::full({batch_size, seq_len, num_head,dim_head_mul3}, -1, qkv.dtype(), qkv.place());

    auto cu_stream = qkv.stream();
    assert(dim_head % 2 == 0); 
    
    dim3 grid(batch_size, seq_len * rotary_emb_dims, num_head);
    const int last_dim = dim_head / rotary_emb_dims;
    auto getBlockSize = [](int dim) {
        if (dim > 256) {
        return 512;
        } else if (dim > 128) {
        return 256;
        } else if (dim > 64) {
        return 128;
        } else if (dim > 32) {
        return 64;
        } else {
        return 32;
        }
    };
    int BlockSize = getBlockSize(last_dim / 2);
    const float *cos_emb = rotary_emb.data<float>();
    const float *sin_emb = rotary_emb.data<float>() + batch_size * seq_len * dim_head;
    const int64_t offset = batch_size * seq_len * num_head * dim_head; 
    const DataType_* qkv_data = reinterpret_cast<const DataType_*>(qkv.data<data_t>()); 

    NeoXRotaryKernel<<<grid, BlockSize, 0, cu_stream>>>(
        qkv_data,
        cos_emb,
        sin_emb,
        nullptr, /*sequence_lengths*/ // TODO(Zhengzekang): Support Variable length. 
        reinterpret_cast<DataType_*>(const_cast<data_t*>(qkv_out.data<data_t>())),
        rotary_emb_dims,
        batch_size,
        seq_len * rotary_emb_dims,
        num_head,
        dim_head, 
        dim_head_mul3, 
        last_dim);
    return {qkv_out};
}

std::vector<paddle::Tensor> RotaryQK(const paddle::Tensor& qkv, 
                                     const paddle::Tensor& rotary_emb) {
    switch (qkv.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchRotaryQK<paddle::DataType::BFLOAT16>(
                qkv, rotary_emb
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchRotaryQK<paddle::DataType::FLOAT16>(
                qkv, rotary_emb
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchRotaryQK<paddle::DataType::FLOAT32>(
                qkv, rotary_emb
            );
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only bfloat16, float16 and float32 are supported. ");
            break;
        }
    }
}


std::vector<std::vector<int64_t>> RotaryQKInferShape(const std::vector<int64_t>& qkv_shape, 
                                                     const std::vector<int64_t>& rotary_emb_shape) {
    const int64_t batch_size = qkv_shape[0]; 
    const int64_t seq_len = qkv_shape[1]; 
    const int64_t num_head = qkv_shape[2]; 
    const int64_t dim_head = qkv_shape[3] / 3; // Since QKV 

    std::vector<int64_t> qkv_out_shape = {batch_size, seq_len, num_head, dim_head * 3};                                                          
    return {qkv_out_shape};
}

std::vector<paddle::DataType> RotaryQKInferDtype(const paddle::DataType& qkv_dtype, 
                                                 const paddle::DataType& rotary_emb_dtype) {
    return {qkv_dtype};
}

PD_BUILD_OP(neox_rope)
    .Inputs({"qkv", "rotary_emb"})
    .Outputs({"qkv_out"})
    .SetKernelFn(PD_KERNEL(RotaryQK))
    .SetInferShapeFn(PD_INFER_SHAPE(RotaryQKInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RotaryQKInferDtype));