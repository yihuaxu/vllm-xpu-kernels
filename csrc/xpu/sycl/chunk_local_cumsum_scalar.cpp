#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include <cmath>
#include <c10/macros/Macros.h>
#include <future>
#include <functional>

template <typename TS, typename TO>
void call_chunk_local_cumsum_scalar(
    TS* s,
    TO* o,
    long* cu_seqlens,
    long* chunk_indices_data,
    const unsigned int T,
    const unsigned int B,
    const unsigned int H,
    const unsigned int BT,
    const bool REVERSE,
    const bool IS_VARLEN,
    const bool HEAD_FIRST,
    const unsigned int NT,
    const unsigned int chunk_indices_len) {
  auto& queue = vllm::xpu::vllmGetQueue();
  sycl::event copy_event;
  long* chunk_indices;
  if (IS_VARLEN) {
    chunk_indices = sycl::malloc_device<long>(chunk_indices_len, queue);
    copy_event = queue.memcpy(
        chunk_indices, chunk_indices_data, chunk_indices_len * sizeof(long));
  }
  sycl::event kernel_event = queue.submit([&](sycl::handler& CGH) {
    if (IS_VARLEN) {
      CGH.depends_on(copy_event);
    }
    CGH.parallel_for(sycl::range<2>{NT, B * H}, [=](sycl::id<2> id) {
      unsigned int i_t = id[0];
      unsigned int i_bh = id[1];
      unsigned int i_b = i_bh / H;
      unsigned int i_h = i_bh % H;
      unsigned int offset;
      unsigned int bos, eos, bot, eot;
      if (IS_VARLEN) {
        long* p_ci = chunk_indices + i_t * 2;
        unsigned int i_n = *p_ci++;
        i_t = *p_ci;
        long* p_cs = cu_seqlens + i_n;
        bos = *p_cs++;
        eos = *p_cs;
        bot = i_t * BT;
        eot = bot + BT;
        if (eot > T) {
          eot = T;
        }
      } else {
        bos = i_b * T;
        eos = bos + T;
        bot = i_t * BT;
        eot = bot + BT;
        if (eot > T) {
          eot = T;
        }
      }
      TS* p_s = s;
      TO* p_o = o;
      unsigned int stride = HEAD_FIRST ? 1 : H;

      if (REVERSE) {
        eot = eot - 1;
        offset = HEAD_FIRST ? bos * H + i_h * T + eot : (bos + eot) * H + i_h;

        p_s += offset;
        p_o += offset;

        float sum = float(*p_s);
        *p_o = TO(sum);
        for (unsigned int t = eot; t > bot; t--) {
          p_s -= stride;
          p_o -= stride;
          sum += float(*p_s);
          *p_o = TO(sum);
        }
      } else {
        offset = HEAD_FIRST ? bos * H + i_h * T + bot : (bos + bot) * H + i_h;

        p_s += offset;
        p_o += offset;

        float sum = float(*p_s);
        *p_o = TO(sum);
        for (unsigned int t = bot + 1; t < eot; t++) {
          p_s += stride;
          p_o += stride;
          sum += float(*p_s);
          *p_o = TO(sum);
        }
      }
    });
  });
  auto future = std::async(std::launch::async, [&]() {
    kernel_event.wait();
    free(chunk_indices, queue);
  });
}

size_t prepare_chunk_indices(
    std::vector<long>& cu_seqlens_vec,
    std::vector<long>& chunk_indices_vec,
    int64_t chunk_size) {
  auto cu_seqlens_len = cu_seqlens_vec.size();
  std::vector<long> indices_vec;
  for (size_t i = 0; i < cu_seqlens_len - 1; i++) {
    auto len = (cu_seqlens_vec[i + 1] - cu_seqlens_vec[i] + chunk_size - 1) /
               chunk_size;
    for (size_t j = 0; j < len; j++) {
      indices_vec.push_back(j);
    }
  }

  size_t len = indices_vec.size();
  chunk_indices_vec.resize(len * 2);
  long sum = 0;
  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i] == 0) sum++;
    chunk_indices_vec[2 * i] = sum - 1;
    chunk_indices_vec[2 * i + 1] = indices_vec[i];
  }

  return len;
}

at::Tensor chunk_local_cumsum_scalar(
    const at::Tensor& g,
    long chunk_size,
    bool reverse,
    const ::std::optional<at::Tensor>& cu_seqlens,
    bool head_first,
    ::std::optional<at::ScalarType> output_dtype) {
  auto shape = g.sizes();
  auto B = shape[0];
  auto T = shape[1];
  auto H = shape[2];
  if (head_first) {
    H = shape[1];
    T = shape[2];
  }
  TORCH_CHECK(
      chunk_size > 0 && (chunk_size & (chunk_size - 1)) == 0,
      "chunk_size must be a power of 2");
  at::ScalarType dtype =
      output_dtype.has_value() ? output_dtype.value() : at::ScalarType::Float;
  auto out = at::zeros_like(g, dtype);
  std::vector<long> cu_seqlens_vec;
  std::vector<long> chunk_indices_vec;
  size_t chunk_indices_len;
  if (cu_seqlens.has_value()) {
    auto cu_seqlens_cpu = cu_seqlens->to(at::kCPU, at::kLong).contiguous();
    auto cu_seqlens_cpu_data =
        reinterpret_cast<long*>(cu_seqlens_cpu.data_ptr());
    cu_seqlens_vec = std::vector<long>(
        cu_seqlens_cpu_data, cu_seqlens_cpu_data + cu_seqlens.value().numel());
    chunk_indices_len =
        prepare_chunk_indices(cu_seqlens_vec, chunk_indices_vec, chunk_size);
  }
  auto BT = chunk_size;
  auto NT = cu_seqlens.has_value() ? chunk_indices_len : (T + BT - 1) / BT;

  bool is_varlen = cu_seqlens.has_value();
  if (output_dtype == torch::kFloat) {
    switch (g.scalar_type()) {
      case torch::kFloat:
        call_chunk_local_cumsum_scalar<float, float>(
            reinterpret_cast<float*>(g.data_ptr()),
            reinterpret_cast<float*>(out.data_ptr()),
            is_varlen ? reinterpret_cast<long*>(cu_seqlens->data_ptr())
                      : nullptr,
            is_varlen ? reinterpret_cast<long*>(chunk_indices_vec.data())
                      : nullptr,
            T,
            B,
            H,
            BT,
            reverse,
            is_varlen,
            head_first,
            NT,
            chunk_indices_vec.size());
        break;
      case torch::kFloat16:
        call_chunk_local_cumsum_scalar<sycl::half, float>(
            reinterpret_cast<sycl::half*>(g.data_ptr()),
            reinterpret_cast<float*>(out.data_ptr()),
            is_varlen ? reinterpret_cast<long*>(cu_seqlens->data_ptr())
                      : nullptr,
            is_varlen ? reinterpret_cast<long*>(chunk_indices_vec.data())
                      : nullptr,
            T,
            B,
            H,
            BT,
            reverse,
            is_varlen,
            head_first,
            NT,
            chunk_indices_vec.size());
        break;
      case torch::kBFloat16:
        call_chunk_local_cumsum_scalar<sycl::ext::oneapi::bfloat16, float>(
            reinterpret_cast<sycl::ext::oneapi::bfloat16*>(g.data_ptr()),
            reinterpret_cast<float*>(out.data_ptr()),
            is_varlen ? reinterpret_cast<long*>(cu_seqlens->data_ptr())
                      : nullptr,
            is_varlen ? reinterpret_cast<long*>(chunk_indices_vec.data())
                      : nullptr,
            T,
            B,
            H,
            BT,
            reverse,
            is_varlen,
            head_first,
            NT,
            chunk_indices_vec.size());
        break;
      default:
        throw std::invalid_argument(
            "Invalid input dtype, only supports float32, float16, and "
            "bfloat16");
        break;
    }
  } else {
    throw std::invalid_argument("Invalid out dtype, only supports float32.");
  }

  return out;
}
