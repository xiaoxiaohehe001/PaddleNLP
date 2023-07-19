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


constexpr int kBlockSize = 256; 
constexpr int kNumWaves = 16; 

inline cudaError_t GetNumBlocks(int64_t n, int* num_blocks) {
    int dev;
    {
      cudaError_t err = cudaGetDevice(&dev);
      if (err != cudaSuccess) { return err; }
    }
    int sm_count;
    {
      cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
      if (err != cudaSuccess) { return err; }
    }
    int tpm;
    {
      cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
      if (err != cudaSuccess) { return err; }
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                                     sm_count * tpm / kBlockSize * kNumWaves));
    return cudaSuccess;
}

template <typename T>
__global__ void VariableLengthNeoXRotaryKernel(const T *input,
                                               const int *padding_offset,
                                               const int *seq_lens,
                                               const float *cos_emb,
                                               const float *sin_emb,
                                               T *output,
                                               const int elem_cnt, 
                                               const int rotary_emb_dims,
                                               const int batch_size,
                                               const int seq_len,
                                               const int token_num, // assume we have batchsize=3, each seq is: [1, 3, 10], and token_num = sum(seq) = 14
                                               const int num_head,
                                               const int dim_head,
                                               const int dim_head_mul3,
                                               const int last_dim) {

    int32_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int32_t hidden_size = num_head * dim_head;
    const int32_t fused_hidden_size = hidden_size * 3;
    const int32_t qk_hidden_size = hidden_size * 2;
    const int32_t qk_head_size = dim_head * 2;

    int half_lastdim = last_dim / 2;
    for (int32_t linear_index = global_thread_idx, step = gridDim.x * blockDim.x; linear_index < elem_cnt; linear_index += step) {
        const int32_t dual_index = linear_index * 2; 
        const int32_t token_idx = dual_index / qk_hidden_size;
        const int32_t ori_token_idx =
            token_idx + (padding_offset == nullptr ? 0 : padding_offset[token_idx]);
        const int32_t target_batch_id = ori_token_idx / seq_len;
        if (seq_lens[target_batch_id] == 0) continue;
        const int32_t seq_id = ori_token_idx % seq_len;

        // qk: token_num, num_head, 2, dim_head
        
        // Equals to: dual_index % (num_head * dim_head * 2) / (dim_head * 2)
        const int32_t head_id = (dual_index % qk_hidden_size) / qk_head_size;
        
        // Equals to: dual_index % (dim_head * 2) / (dim_head)
        const int32_t qkv_id = (dual_index % qk_head_size) / dim_head; // To choose in Q or K. 

        const int32_t size_id = linear_index % (dim_head / 2); // since each thread process 2 values. 
        printf("linearidx: %d, tokenidx: %d, oritokenidx: %d, target_batchidx: %d, seq_id: %d, head_id: %d, qkv_id is: %d, size_id is: %d \n", linear_index, token_idx, ori_token_idx, target_batch_id, seq_id, head_id, qkv_id, size_id); 

        // Equals to: token_idx * num_head * 3 * dim_head + head_id * 3 * dim_head + qkv_idx * dim_head + size_id; 
        int base_idx = token_idx * fused_hidden_size +
                       head_id * dim_head_mul3 + 
                       qkv_id * dim_head + size_id; 

        int left_idx = base_idx;
        const int right_idx = base_idx + half_lastdim;
        int emb_idx_left = target_batch_id * seq_len * last_dim + seq_id * last_dim + size_id;
        int emb_idx_right = emb_idx_left + half_lastdim; 


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


template <paddle::DataType D>
std::vector<paddle::Tensor> LaunchRotaryQK(const paddle::Tensor& qkv, 
                                           const paddle::Tensor& rotary_emb, 
                                           const paddle::Tensor& padding_offset, 
                                           const paddle::Tensor& seq_lens) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int rotary_emb_dims = 1; // Since LLAMA Neox RotaryEmbedding no need rotary_emb_dims. Author(Zhengzekang)
    // rotary_emb shape: batchsize * 2, 1, maxseq_length, head_size
    const int64_t batch_size = rotary_emb.shape()[0] / 2; // since cos_emb and sin_emb is stack in axis0. 
    const int64_t seq_len = rotary_emb.shape()[2]; 
    const int64_t token_num = qkv.shape()[0]; 
    const int64_t num_head = qkv.shape()[1]; 
    const int64_t dim_head_mul3 = qkv.shape()[2]; 
    const int64_t dim_head = dim_head_mul3 / 3; 
    const int last_dim = dim_head / rotary_emb_dims;

    const int64_t elem_cnt = token_num * num_head * dim_head * 2 / 2; // mul2 is for process Q and K, divide2 is each threads process 2val. 

    auto tmp_out = paddle::full({1}, -1, qkv.dtype(), qkv.place());

    auto cu_stream = qkv.stream();
    assert(dim_head % 2 == 0); 
    
    int32_t grid_size = 1; 
    GetNumBlocks(elem_cnt, &grid_size); 

    const float *cos_emb = rotary_emb.data<float>();
    const float *sin_emb = rotary_emb.data<float>() + batch_size * seq_len * dim_head;
    const int64_t offset = batch_size * seq_len * num_head * dim_head; 
    const DataType_* qkv_data = reinterpret_cast<const DataType_*>(qkv.data<data_t>()); 

    const int32_t* padding_offset_data = reinterpret_cast<const int32_t*>(padding_offset.data<int32_t>()); 
    const int32_t* seq_lens_data = reinterpret_cast<const int32_t*>(seq_lens.data<int32_t>()); 

    VariableLengthNeoXRotaryKernel<<<grid_size, kBlockSize, 0, cu_stream>>>(
        qkv_data,
        padding_offset_data, 
        seq_lens_data, 
        cos_emb,
        sin_emb,
        reinterpret_cast<DataType_*>(const_cast<data_t*>(qkv.data<data_t>())),
        elem_cnt, 
        rotary_emb_dims,
        batch_size,
        seq_len * rotary_emb_dims,
        token_num, 
        num_head,
        dim_head, 
        dim_head_mul3, 
        last_dim);
    return {tmp_out};
}

std::vector<paddle::Tensor> RotaryQK(const paddle::Tensor& qkv, 
                                     const paddle::Tensor& rotary_emb, 
                                     const paddle::Tensor& padding_offset, 
                                     const paddle::Tensor& seq_lens) {
    switch (qkv.type()) {
        case paddle::DataType::BFLOAT16: {
            return LaunchRotaryQK<paddle::DataType::BFLOAT16>(
                qkv, rotary_emb, padding_offset, seq_lens
            );
        }
        case paddle::DataType::FLOAT16: {
            return LaunchRotaryQK<paddle::DataType::FLOAT16>(
                qkv, rotary_emb, padding_offset, seq_lens
            );
        }
        case paddle::DataType::FLOAT32: {
            return LaunchRotaryQK<paddle::DataType::FLOAT32>(
                qkv, rotary_emb, padding_offset, seq_lens
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
                                                     const std::vector<int64_t>& rotary_emb_shape, 
                                                     const std::vector<int64_t>& padding_offset_shape, 
                                                     const std::vector<int64_t>& seq_lens_shape) {
    std::vector<int64_t> tmp_out_shape = {1};                                                          
    return {tmp_out_shape};
}

std::vector<paddle::DataType> RotaryQKInferDtype(const paddle::DataType& qkv_dtype, 
                                                 const paddle::DataType& rotary_emb_dtype, 
                                                 const paddle::DataType& padding_offset_dtype, 
                                                 const paddle::DataType& seq_lens_dtype) {
    return {qkv_dtype};
}

PD_BUILD_OP(neox_rope)
    .Inputs({"qkv", "rotary_emb", "seq_lens", "padding_offset"})
    .Outputs({"tmp_out"})
    .SetKernelFn(PD_KERNEL(RotaryQK))
    .SetInferShapeFn(PD_INFER_SHAPE(RotaryQKInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RotaryQKInferDtype));