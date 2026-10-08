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

// The Tensor API relation head against the PyTorch reference.
//
// Authors the "relations" signature from relhead.safetensors, compiles it with
// LiteRT (CPU by default, --accelerator=gpu for Metal / the GPU accelerator),
// runs every case of relhead_fixture.safetensors (ram/export_tensorapi.py: 4
// six-object PSG test images + 16 cases with 2..5 objects in 6 slots) and
// compares the logits of every ordered pair of real objects with PyTorch.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "gtest/gtest.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "relhead_graph.h"
#include "relhead_runner.h"
#include "sam2_pipeline.h"
#include "signature_stage.h"
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/examples/gemma3/safetensors.h"

namespace sam2_chain {
namespace {

std::string g_weights, g_fixture, g_accelerator = "cpu", g_precision = "fp16";

using ::litert::tensor::examples::SafetensorLoader;

// Raw bytes of one fixture tensor.
std::vector<uint8_t> Bytes(const SafetensorLoader& f, const std::string& name) {
  auto info = f.GetTensorInfo(name);
  EXPECT_TRUE(info.ok()) << name;
  if (!info.ok()) return {};
  const std::byte* p = info->storage->data_base + info->data_start;
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(p),
                              reinterpret_cast<const uint8_t*>(p) + (info->data_end - info->data_start));
}
std::vector<float> Floats(const SafetensorLoader& f, const std::string& name) {
  std::vector<uint8_t> b = Bytes(f, name);
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), v.size() * 4);
  return v;
}

TEST(RelHeadGraph, MatchesPyTorch) {
  ASSERT_FALSE(g_weights.empty()) << "--weights";
  ASSERT_FALSE(g_fixture.empty()) << "--fixture";
  auto weights = LoadRelHeadWeights(g_weights);
  ASSERT_TRUE(weights.ok()) << weights.status();
  auto fixture = SafetensorLoader::Load(g_fixture);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  int cases = 0;
  while (fixture->GetTensorInfo(absl::StrCat("img", cases, "_masks")).ok()) ++cases;
  ASSERT_GT(cases, 0);
  auto m0 = fixture->GetTensorInfo("img0_masks");
  const int side = static_cast<int>(m0->shape[1]);
  std::cout << "relation head: d " << weights->d << ", " << weights->layers << " layers, "
            << weights->classes << " classes, " << weights->preds << " predicates; masks "
            << side << "x" << side << "; " << cases << " cases" << std::endl;

  // ---- Author + serialize + compile through RelationRunner (what the wasm
  // build and the pipeline use).
  auto env_or = litert::Environment::Create({});
  ASSERT_TRUE(env_or);
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));
  ModelCompiler compile = [&](const std::string& path)
      -> absl::StatusOr<std::shared_ptr<litert::CompiledModel>> {
    auto options = litert::Options::Create();
    if (!options) return absl::InternalError("Options::Create failed");
    options->SetHardwareAccelerators(g_accelerator == "gpu" ? litert::HwAccelerators::kGpu
                                                            : litert::HwAccelerators::kCpu);
    if (g_accelerator == "gpu" && g_precision == "fp32") {
      if (auto gpu = options->GetGpuOptions()) gpu->SetPrecision(litert::GpuOptions::Precision::kFp32);
    }
    auto model = litert::CompiledModel::Create(*env, path, *options);
    if (!model) return absl::InternalError(model.Error().Message());
    return std::make_shared<litert::CompiledModel>(std::move(*model));
  };
  auto runner_or = RelationRunner::Create(env, compile, *weights, side, ::testing::TempDir());
  ASSERT_TRUE(runner_or.ok()) << runner_or.status();
  RelationRunner& runner = **runner_or;

  std::vector<std::shared_ptr<LitertBuffer>> buffers(kMaxObjects);
  for (auto& b : buffers) {
    auto buf = runner.AllocateMask();
    ASSERT_TRUE(buf.ok()) << buf.status();
    b = *buf;
  }
  const int P = weights->preds, S2 = side * side;

  double worst = 0, total_ms = 0;
  int top1_same = 0, pairs = 0;
  for (int c = 0; c < cases; ++c) {
    const std::string pre = absl::StrCat("img", c, "_");
    std::vector<uint8_t> bits = Bytes(*fixture, pre + "masks");
    std::vector<float> cls = Floats(*fixture, pre + "cls"), valid = Floats(*fixture, pre + "valid");
    std::vector<float> ref = Floats(*fixture, pre + "logits");
    ASSERT_EQ(bits.size(), static_cast<size_t>(kMaxObjects) * S2);
    std::array<std::shared_ptr<LitertBuffer>, kMaxObjects> masks{};
    std::array<int, kMaxObjects> classes{};
    for (int k = 0; k < kMaxObjects; ++k) {
      if (valid[k] <= 0) continue;  // empty slot: the runner binds its -1024 mask
      // Logits as SAM 2 writes them: positive inside, negative outside.
      std::vector<float> m(S2);
      for (int i = 0; i < S2; ++i) m[i] = bits[k * S2 + i] ? 4.f : -4.f;
      ASSERT_TRUE(WriteFloats(*buffers[k], m).ok());
      masks[k] = buffers[k];
      classes[k] = static_cast<int>(cls[k]);
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto out = runner.Run(masks, classes);
    total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ASSERT_TRUE(out.ok()) << out.status();
    ASSERT_EQ(out->size(), static_cast<size_t>(kMaxObjects) * kMaxObjects * P);

    int n = 0;
    for (float v : valid) n += v > 0;
    double diff = 0;
    for (int s = 0; s < n; ++s) {
      for (int o = 0; o < n; ++o) {
        if (s == o) continue;
        for (int p = 0; p < P; ++p) {
          const size_t i = (static_cast<size_t>(s) * kMaxObjects + o) * P + p;
          diff = std::max(diff, static_cast<double>(std::fabs((*out)[i] - ref[i])));
        }
        const size_t b0 = (static_cast<size_t>(s) * kMaxObjects + o) * P;
        top1_same += std::max_element(out->begin() + b0, out->begin() + b0 + P) - (out->begin() + b0) ==
                     std::max_element(ref.begin() + b0, ref.begin() + b0 + P) - (ref.begin() + b0);
        ++pairs;
      }
    }
    worst = std::max(worst, diff);
    std::cout << "  case " << c << ": " << n << " objects, max |tensor api - torch| = " << diff << std::endl;
  }
  std::cout << "relations (" << g_accelerator << (g_accelerator == "gpu" ? " " + g_precision : "")
            << "): worst " << worst << ", top-1 predicate agrees on " << top1_same << "/" << pairs
            << " pairs, mean "
            << total_ms / cases << " ms per run (incl. readback)" << std::endl;
  // fp32 CPU: ~1e-5 (the numpy model of this graph is 2e-5 off). The GPU
  // accelerator may run fp16.
  EXPECT_LT(worst, g_accelerator == "gpu" && g_precision == "fp16" ? 0.25 : 1e-3);
  EXPECT_GE(top1_same, pairs * 0.97);
}

}  // namespace
}  // namespace sam2_chain

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--weights=", 0) == 0) sam2_chain::g_weights = a.substr(10);
    if (a.rfind("--fixture=", 0) == 0) sam2_chain::g_fixture = a.substr(10);
    if (a.rfind("--accelerator=", 0) == 0) sam2_chain::g_accelerator = a.substr(14);
    if (a.rfind("--gpu_precision=", 0) == 0) sam2_chain::g_precision = a.substr(16);
  }
  return RUN_ALL_TESTS();
}
