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

#include "relhead_runner.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace sam2_chain {

absl::StatusOr<std::unique_ptr<RelationRunner>> RelationRunner::Create(
    std::shared_ptr<litert::Environment> env, const ModelCompiler& compile,
    const RelHeadWeights& weights, int mask_side, const std::string& scratch_dir) {
  std::unique_ptr<RelationRunner> r(new RelationRunner());
  r->preds_ = weights.preds;
  r->classes_ = weights.classes;
  {
    ModelFactory factory;
    if (auto st = AddRelationSignature(factory, weights, mask_side); !st.ok()) return st;
    const std::string path = absl::StrCat(scratch_dir, "/relhead_", mask_side, ".tflite");
    if (auto st = factory.Save(path); !st.ok()) return st;
    auto model = compile(path);
    if (!model.ok()) return model.status();
    r->model_ = *std::move(model);
  }
  auto stage = SignatureStage::Create("relations", r->model_, kRelSignature);
  if (!stage.ok()) return stage.status();
  r->stage_ = *std::move(stage);
  r->stage_->SetEnvironment(env);
  r->env_ = env;

  auto alloc = [&](const std::string& name) -> absl::StatusOr<std::shared_ptr<LitertBuffer>> {
    auto desc = r->stage_->GetInputDescriptor(name);
    if (!desc.ok()) return desc.status();
    return AllocateLike(env, *desc);
  };
  auto empty = alloc(MaskInput(0));
  if (!empty.ok()) return empty.status();
  r->empty_mask_ = *empty;
  if (auto st = WriteFloats(*r->empty_mask_,
                            std::vector<float>(static_cast<size_t>(mask_side) * mask_side, -1024.f));
      !st.ok()) {
    return st;
  }
  auto cls = alloc(kRelClsInput);
  if (!cls.ok()) return cls.status();
  r->cls_ = *cls;
  auto valid = alloc(kRelValidInput);
  if (!valid.ok()) return valid.status();
  r->valid_ = *valid;
  if (auto st = r->stage_->SetInputBuffer(kRelClsInput, r->cls_); !st.ok()) return st;
  if (auto st = r->stage_->SetInputBuffer(kRelValidInput, r->valid_); !st.ok()) return st;
  return r;
}

absl::StatusOr<std::shared_ptr<LitertBuffer>> RelationRunner::AllocateMask() const {
  auto desc = stage_->GetInputDescriptor(MaskInput(0));
  if (!desc.ok()) return desc.status();
  return AllocateLike(env_, *desc);
}

absl::StatusOr<std::vector<float>> RelationRunner::Run(
    const std::array<std::shared_ptr<LitertBuffer>, kMaxObjects>& masks,
    const std::array<int, kMaxObjects>& cls) {
  std::array<float, kMaxObjects> valid{};
  for (int k = 0; k < kMaxObjects; ++k) {
    valid[k] = masks[k] ? 1.f : 0.f;
    if (auto st = stage_->SetInputBuffer(MaskInput(k), masks[k] ? masks[k] : empty_mask_);
        !st.ok()) {
      return st;
    }
  }
  // Classes and slot occupancy change rarely: write them only when they do.
  if (!written_ || cls != last_cls_) {
    std::vector<float> onehot(static_cast<size_t>(kMaxObjects) * classes_, 0.f);
    for (int k = 0; k < kMaxObjects; ++k) {
      const int c = cls[k] >= 0 && cls[k] < classes_ ? cls[k] : 0;
      onehot[static_cast<size_t>(k) * classes_ + c] = 1.f;
    }
    if (auto st = WriteFloats(*cls_, onehot); !st.ok()) return st;
    last_cls_ = cls;
  }
  if (!written_ || valid != last_valid_) {
    if (auto st = WriteFloats(*valid_, std::vector<float>(valid.begin(), valid.end())); !st.ok()) {
      return st;
    }
    last_valid_ = valid;
  }
  written_ = true;
  if (auto st = stage_->Run(); !st.ok()) return st;
  return ReadFloats(*stage_->GetOutputBuffer(kRelLogitsOutput));
}

absl::StatusOr<std::vector<float>> RelationRunner::Run(
    const Sam2Pipeline& pipeline, int t, const std::array<int, kMaxObjects>& cls) {
  std::array<std::shared_ptr<LitertBuffer>, kMaxObjects> masks{};
  for (int k = 0; k < kMaxObjects; ++k) {
    if (const FrameResult* r = pipeline.Result(k, t)) masks[k] = r->low_mask;
  }
  return Run(masks, cls);
}

}  // namespace sam2_chain
