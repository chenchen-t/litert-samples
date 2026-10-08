// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "relhead_graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/examples/gemma3/safetensors.h"
#include "tensor/tensor.h"

namespace sam2_chain {
namespace {

using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::examples::SafetensorLoader;
using ::litert::tensor::examples::SafetensorTensorInfo;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::Type;

TfTensor Const(std::vector<float> values, std::vector<int> shape) {
  return TfTensor({.type = Type::kFP32,
                   .shape = std::move(shape),
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)});
}
TfTensor Scalar(float v) { return Const({v}, {1}); }
TfTensor Input(const std::string& name, std::vector<int> shape) {
  return TfTensor({.name = name, .type = Type::kFP32, .shape = std::move(shape)});
}

// x > 0 as {0, 1} with Relu / Mul / Sub only (as sam2v_graph's StepPos): no
// Greater / Cast, so the signature stays in one WebGPU partition. Equal to
// x > 0 except for 0 < x < 1e-4.
TfTensor StepPos(const TfTensor& x) {
  const TfTensor one = Scalar(1.f);
  return Sub(one, Relu(Sub(one, Mul(Relu(x), Scalar(1e4f)))));
}

// Rows [r0, r0 + n) of a row-major [rows, cols] weight.
std::vector<float> Rows(const RelHeadWeights::Entry& e, int r0, int n) {
  const int cols = e.shape[1];
  return {e.values.begin() + static_cast<size_t>(r0) * cols,
          e.values.begin() + static_cast<size_t>(r0 + n) * cols};
}
// Columns [c0, c0 + n) of a row-major [rows, cols] weight, as [rows, n].
std::vector<float> Cols(const RelHeadWeights::Entry& e, int c0, int n) {
  const int rows = e.shape[0], cols = e.shape[1];
  std::vector<float> out(static_cast<size_t>(rows) * n);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < n; ++c) out[r * n + c] = e.values[r * cols + c0 + c];
  }
  return out;
}

class Builder {
 public:
  explicit Builder(const RelHeadWeights& w) : w_(w) {}

  TfTensor W(const std::string& name) const {
    const auto& e = w_.at(name);
    return Const(e.values, e.shape);
  }
  // nn.Linear: x [..., in] -> [..., out].
  TfTensor Linear(const TfTensor& x, const std::string& p) const {
    return FullyConnected(x, W(p + ".weight"), W(p + ".bias"));
  }
  TfTensor LayerNorm(const TfTensor& x, const std::string& p) const {
    const int last = static_cast<int>(x.GetShape().size()) - 1;
    TfTensor c = Sub(x, Mean(x, {last}, /*keep_dims=*/true));
    TfTensor var = Mean(Square(c), {last}, /*keep_dims=*/true);
    TfTensor n = Mul(c, Rsqrt(Add(var, Scalar(1e-5f))));
    return Add(Mul(n, W(p + ".weight")), W(p + ".bias"));
  }

  // One pre-norm nn.TransformerEncoderLayer (ReLU feed-forward) over the K
  // slots; `key_bias` [1, 1, K] is 0 for objects and -1e4 for empty slots.
  TfTensor EncoderLayer(const TfTensor& x, const std::string& p,
                        const TfTensor& key_bias) const {
    const int d = w_.d, H = w_.heads, hd = d / H;
    // in_proj with 1/sqrt(hd) folded into the q rows.
    const auto& win = w_.at(p + ".self_attn.in_proj_weight");
    const auto& bin = w_.at(p + ".self_attn.in_proj_bias");
    const float scale = 1.f / std::sqrt(static_cast<float>(hd));
    std::vector<float> wq = win.values, bq = bin.values;
    for (size_t i = 0; i < static_cast<size_t>(d) * d; ++i) wq[i] *= scale;
    for (int i = 0; i < d; ++i) bq[i] *= scale;
    TfTensor xn = LayerNorm(x, p + ".norm1");
    TfTensor qkv = FullyConnected(xn, Const(wq, {3 * d, d}), Const(bq, {3 * d}));
    auto heads = [&](int i) {  // [K, d] slice i of q / k / v -> [H, K, hd]
      TfTensor t = Slice(qkv, {0, i * d}, {kMaxObjects, d});
      return Transpose(Reshape(t, {kMaxObjects, H, hd}), {1, 0, 2});
    };
    TfTensor q = heads(0), k = heads(1), v = heads(2);
    TfTensor att = Softmax(Add(BatchMatMul(q, k, false, true), key_bias));
    TfTensor o = Reshape(Transpose(BatchMatMul(att, v), {1, 0, 2}), {kMaxObjects, d});
    TfTensor y = Add(x, Linear(o, p + ".self_attn.out_proj"));
    TfTensor ff = Linear(Relu(Linear(LayerNorm(y, p + ".norm2"), p + ".linear1")),
                         p + ".linear2");
    return Add(y, ff);
  }

 private:
  const RelHeadWeights& w_;
};

absl::StatusOr<std::vector<float>> ReadFp32(const SafetensorTensorInfo& info) {
  const auto* storage = info.storage.get();
  if (storage == nullptr || storage->data_base == nullptr) {
    return absl::FailedPreconditionError(absl::StrCat(info.name, ": invalid storage"));
  }
  if (info.dtype != ::safetensors::dtype::kFLOAT32) {
    return absl::InvalidArgumentError(absl::StrCat(info.name, ": expected F32"));
  }
  size_t count = 1;
  for (int64_t d : info.shape) count *= static_cast<size_t>(d);
  if (info.data_end - info.data_start != count * sizeof(float) ||
      info.data_end > storage->data_size) {
    return absl::DataLossError(absl::StrCat(info.name, ": size mismatch"));
  }
  std::vector<float> out(count);
  std::memcpy(out.data(), storage->data_base + info.data_start, count * sizeof(float));
  return out;
}

}  // namespace

const RelHeadWeights::Entry& RelHeadWeights::at(const std::string& name) const {
  auto it = tensors.find(name);
  ABSL_CHECK(it != tensors.end()) << "relation head: missing tensor " << name;
  return it->second;
}

absl::StatusOr<RelHeadWeights> LoadRelHeadWeights(const std::string& path) {
  auto loader = SafetensorLoader::Load(path);
  if (!loader.ok()) return loader.status();
  RelHeadWeights w;
  for (const std::string& name : loader->GetTensorNames()) {
    auto info = loader->GetTensorInfo(name);
    if (!info.ok()) return info.status();
    auto values = ReadFp32(*info);
    if (!values.ok()) return values.status();
    w.tensors[name] = {std::vector<int>(info->shape.begin(), info->shape.end()),
                       *std::move(values)};
  }
  for (const char* need : {"emb.weight", "inp.0.weight", "head.0.weight", "head.3.weight", "pg.0.weight"}) {
    if (!w.tensors.contains(need)) {
      return absl::NotFoundError(absl::StrCat(path, ": missing ", need));
    }
  }
  w.d = w.tensors["inp.0.weight"].shape[0];
  w.classes = w.tensors["emb.weight"].shape[0];
  w.preds = w.tensors["head.3.weight"].shape[0];
  while (w.tensors.contains(absl::StrCat("enc.layers.", w.layers, ".norm1.weight"))) ++w.layers;
  if (w.d % w.heads != 0 || w.layers == 0) {
    return absl::InvalidArgumentError(absl::StrCat(path, ": unexpected relation head shapes"));
  }
  return w;
}

absl::Status AddRelationSignature(ModelFactory& factory, const RelHeadWeights& w,
                                  int mask_side) {
  constexpr int K = kMaxObjects;
  const int side = mask_side, P = side * side, d = w.d;
  const float fside = static_cast<float>(side);
  Builder b(w);

  // ---- Inputs.
  std::vector<TfTensor> masks;
  for (int k = 0; k < K; ++k) masks.push_back(Input(MaskInput(k), {1, side, side}));
  TfTensor cls = Input(kRelClsInput, {K, w.classes});
  TfTensor valid = Input(kRelValidInput, {1, 1, K});

  // ---- Per-object mask geometry (relhead.ts maskBox).
  TfTensor bits = StepPos(Concatenation(absl::MakeSpan(masks), 0));  // [K,side,side]
  TfTensor flat = Reshape(bits, {K, P});
  TfTensor area = Mean(flat, {1}, /*keep_dims=*/true);              // [K,1]
  TfTensor count = Mul(area, Scalar(static_cast<float>(P)));
  TfTensor nonempty = Minimum(count, Scalar(1.f));
  TfTensor rows = ReduceMax(bits, {2}, /*keep_dims=*/false);         // [K,side] any in row y
  TfTensor cols = ReduceMax(bits, {1}, /*keep_dims=*/false);         // [K,side] any in col x
  std::vector<float> up(side), down(side);
  for (int i = 0; i < side; ++i) {
    up[i] = static_cast<float>(i + 1);    // last occupied index + 1
    down[i] = static_cast<float>(side - i);  // side - first occupied index
  }
  const TfTensor ramp_up = Const(up, {side}), ramp_down = Const(down, {side});
  auto hi = [&](const TfTensor& occ) {  // (last + 1) / side, 0 if empty
    return Div(ReduceMax(Mul(occ, ramp_up), {1}, true), Scalar(fside));
  };
  auto lo = [&](const TfTensor& occ) {  // first / side, 0 if empty
    TfTensor m = ReduceMax(Mul(occ, ramp_down), {1}, true);
    return Mul(Div(Sub(Scalar(fside), m), Scalar(fside)), nonempty);
  };
  TfTensor x0 = lo(cols), x1 = hi(cols), y0 = lo(rows), y1 = hi(rows);  // [K,1]
  TfTensor cx = Mul(Add(x0, x1), Scalar(0.5f)), cy = Mul(Add(y0, y1), Scalar(0.5f));
  TfTensor bw = Sub(x1, x0), bh = Sub(y1, y0);

  // ---- Object tokens: [geom 9, class embedding] -> Linear -> LayerNorm -> GELU.
  const auto& emb = w.at("emb.weight");  // [C, 64]
  std::vector<float> emb_t(emb.values.size());
  const int C = emb.shape[0], E = emb.shape[1];
  for (int c = 0; c < C; ++c) {
    for (int e = 0; e < E; ++e) emb_t[e * C + c] = emb.values[c * E + e];
  }
  TfTensor cls_emb = FullyConnected(cls, Const(emb_t, {E, C}));  // [K,64]
  std::vector<TfTensor> parts = {x0, y0, x1, y1, cx, cy, bw, bh, area, cls_emb};
  TfTensor feat = Concatenation(absl::MakeSpan(parts), 1);
  TfTensor x = Gelu(b.LayerNorm(b.Linear(feat, "inp.0"), "inp.1"));    // [K,d]

  // ---- Transformer over the slots; empty slots (no object, or an object
  // whose mask is empty on this frame) are not attended to.
  TfTensor present = Mul(valid, Reshape(nonempty, {1, 1, K}));
  TfTensor key_bias = Mul(Sub(present, Scalar(1.f)), Scalar(1e4f));
  for (int l = 0; l < w.layers; ++l) {
    x = b.EncoderLayer(x, absl::StrCat("enc.layers.", l), key_bias);
  }

  // ---- Pair features [K,K,8] for (s, o) (relhead.ts pairGeom).
  auto row = [&](const TfTensor& t) { return Reshape(t, {1, K}); };  // o axis
  TfTensor w_ = Maximum(bw, Scalar(1e-3f)), h_ = Maximum(bh, Scalar(1e-3f));
  TfTensor sz = Sqrt(Mul(w_, h_));
  TfTensor ref = Mul(Add(sz, row(sz)), Scalar(0.5f));
  TfTensor dx = Div(Sub(row(cx), cx), ref);
  TfTensor dy = Div(Sub(row(cy), cy), ref);
  TfTensor logw = Log(w_), logh = Log(h_);
  TfTensor lw = Sub(row(logw), logw), lh = Sub(row(logh), logh);
  TfTensor ix = Relu(Sub(Minimum(x1, row(x1)), Maximum(x0, row(x0))));
  TfTensor iy = Relu(Sub(Minimum(y1, row(y1)), Maximum(y0, row(y0))));
  TfTensor bi = Mul(ix, iy);
  TfTensor ar = Mul(w_, h_);
  TfTensor biou = Div(bi, Maximum(Sub(Add(ar, row(ar)), bi), Scalar(1e-6f)));
  // Two Reshapes: a GPU delegate rejects a node consuming one value twice.
  TfTensor m3a = Reshape(flat, {1, K, P}), m3b = Reshape(flat, {1, K, P});
  TfTensor inter = Reshape(BatchMatMul(m3a, m3b, false, true), {K, K});
  TfTensor miou = Div(inter, Maximum(Sub(Add(count, row(count)), inter), Scalar(1e-6f)));
  TfTensor cs = Div(inter, Maximum(count, Scalar(1e-6f)));
  TfTensor co = Div(inter, Maximum(row(count), Scalar(1e-6f)));
  std::vector<TfTensor> pf;
  for (const TfTensor& t : {dx, dy, lw, lh, biou, miou, cs, co}) {
    pf.push_back(Reshape(t, {K, K, 1}));
  }
  TfTensor pair = Reshape(Concatenation(absl::MakeSpan(pf), 2), {K * K, 8});

  // ---- Pair head: head.0 on [tok_s, tok_o, GELU(pg(pair))], split by input
  // block so the token parts run once per slot.
  const auto& h0 = w.at("head.0.weight");  // [D, 2d + 64]
  const int D = h0.shape[0], G = h0.shape[1] - 2 * d;
  TfTensor a = FullyConnected(x, Const(Cols(h0, 0, d), {D, d}));       // [K,D]
  TfTensor bo = FullyConnected(x, Const(Cols(h0, d, d), {D, d}));      // [K,D]
  TfTensor g = FullyConnected(Gelu(b.Linear(pair, "pg.0")), Const(Cols(h0, 2 * d, G), {D, G}));                                        // [K*K,D]
  TfTensor hid = Add(Add(Reshape(a, {K, 1, D}), Reshape(bo, {1, K, D})),
                     Add(Reshape(g, {K, K, D}), b.W("head.0.bias")));
  TfTensor logits = b.Linear(Reshape(Gelu(hid), {K * K, D}), "head.3");          // [K*K,P]
  TfTensor out = Reshape(logits, {K, K, w.preds});
  out.SetName(kRelLogitsOutput);

  std::vector<TensorHandle> ins(masks.begin(), masks.end());
  ins.push_back(cls);
  ins.push_back(valid);
  return factory.AddSignature(ins, {TensorHandle(out)}, kRelSignature);
}

}  // namespace sam2_chain
