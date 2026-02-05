#include <sycl/sycl.hpp>
#include "utils.h"
#include "dispatch_utils.h"
#include <cmath>
#include <c10/macros/Macros.h>

template <typename TS, typename TO>
void call_chunk_local_cumsum_scalar(
    TS* s,
    TO* o,
    long* cu_seqlens,
    long* chunk_indices,
    const unsigned int T,
    const unsigned int B,
    const unsigned int H,
    const unsigned int BT,
    const bool REVERSE,
    const bool IS_VARLEN,
    const bool HEAD_FIRST,
    const unsigned int NT) {
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& CGH) {
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
}

at::Tensor chunk_local_cumsum_scalar_kernel(
    const at::Tensor& g,
    at::Tensor& out,
    const ::std::optional<at::Tensor>& cu_seqlens,
    const ::std::optional<at::Tensor>& chunk_indices,
    int64_t T,
    int64_t B,
    int64_t H,
    int64_t BT,
    bool reverse,
    bool head_first,
    int64_t NT) {
  bool is_varlen = cu_seqlens.has_value() && chunk_indices.has_value();
  if (out.scalar_type() == torch::kFloat) {
    switch (g.scalar_type()) {
      case torch::kFloat:
        call_chunk_local_cumsum_scalar<float, float>(
            reinterpret_cast<float*>(g.data_ptr()),
            reinterpret_cast<float*>(out.data_ptr()),
            is_varlen ? reinterpret_cast<long*>(cu_seqlens->data_ptr())
                      : nullptr,
            is_varlen ? reinterpret_cast<long*>(chunk_indices->data_ptr())
                      : nullptr,
            T,
            B,
            H,
            BT,
            reverse,
            is_varlen,
            head_first,
            NT);
        break;
      case torch::kFloat16:
        call_chunk_local_cumsum_scalar<sycl::half, float>(
            reinterpret_cast<sycl::half*>(g.data_ptr()),
            reinterpret_cast<float*>(out.data_ptr()),
            is_varlen ? reinterpret_cast<long*>(cu_seqlens->data_ptr())
                      : nullptr,
            is_varlen ? reinterpret_cast<long*>(chunk_indices->data_ptr())
                      : nullptr,
            T,
            B,
            H,
            BT,
            reverse,
            is_varlen,
            head_first,
            NT);
        break;
      case torch::kBFloat16:
        call_chunk_local_cumsum_scalar<sycl::ext::oneapi::bfloat16, float>(
            reinterpret_cast<sycl::ext::oneapi::bfloat16*>(g.data_ptr()),
            reinterpret_cast<float*>(out.data_ptr()),
            is_varlen ? reinterpret_cast<long*>(cu_seqlens->data_ptr())
                      : nullptr,
            is_varlen ? reinterpret_cast<long*>(chunk_indices->data_ptr())
                      : nullptr,
            T,
            B,
            H,
            BT,
            reverse,
            is_varlen,
            head_first,
            NT);
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
