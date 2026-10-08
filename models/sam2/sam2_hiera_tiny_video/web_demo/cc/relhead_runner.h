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

// Runs the "relations" signature (relhead_graph.h) on the low-res mask
// buffers Sam2Pipeline keeps per object and frame: the masks are bound as
// they are (no copies, no readback); only the [6, 6, P] logits come back.

#ifndef SAM2_WEB_TENSORAPI_CC_RELHEAD_RUNNER_H_
#define SAM2_WEB_TENSORAPI_CC_RELHEAD_RUNNER_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "relhead_graph.h"
#include "sam2_pipeline.h"
#include "signature_stage.h"

namespace sam2_chain {

class RelationRunner {
 public:
  // Authors the relation model for masks of side `mask_side`, serializes it
  // to `scratch_dir` and compiles it with `compile`.
  static absl::StatusOr<std::unique_ptr<RelationRunner>> Create(
      std::shared_ptr<litert::Environment> env, const ModelCompiler& compile,
      const RelHeadWeights& weights, int mask_side, const std::string& scratch_dir);

  int predicates() const { return preds_; }
  int classes() const { return classes_; }

  // A buffer shaped like the mask inputs (tests, hosts without a pipeline).
  absl::StatusOr<std::shared_ptr<LitertBuffer>> AllocateMask() const;

  // masks[k]: slot k's low-res mask logits (null = no object); cls[k]: its
  // class index (0 = unknown). Returns logits [6, 6, P] (s, o, predicate).
  absl::StatusOr<std::vector<float>> Run(
      const std::array<std::shared_ptr<LitertBuffer>, kMaxObjects>& masks,
      const std::array<int, kMaxObjects>& cls);

  // Convenience: the masks of frame t of `pipeline`'s objects.
  absl::StatusOr<std::vector<float>> Run(const Sam2Pipeline& pipeline, int t,
                                         const std::array<int, kMaxObjects>& cls);

 private:
  RelationRunner() = default;

  std::shared_ptr<litert::CompiledModel> model_;
  std::shared_ptr<SignatureStage> stage_;
  std::shared_ptr<LitertBuffer> empty_mask_, cls_, valid_;
  std::array<int, kMaxObjects> last_cls_{};
  std::array<float, kMaxObjects> last_valid_{};
  std::shared_ptr<litert::Environment> env_;
  bool written_ = false;
  int preds_ = 0, classes_ = 0;
};

}  // namespace sam2_chain

#endif  // SAM2_WEB_TENSORAPI_CC_RELHEAD_RUNNER_H_
