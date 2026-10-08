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

// The learned relation head (ram/train_head.py, variant geom_label) as a
// Tensor API graph: one signature, "relations", that reads the SAM 2 low-res
// masks the track steps write and returns predicate logits for every ordered
// pair of the kMaxObjects slots.
//
//   inputs   mask_0 .. mask_5  [1, S/4, S/4]  mask logits (same names and
//                                             shapes as the composite's)
//            rel_cls           [6, C]         one-hot PSG class per slot
//                                             (column 0 = unknown)
//            rel_valid         [1, 1, 6]      1 = slot holds an object
//   output   rel_logits        [6, 6, P]      logits (s, o, predicate);
//                                             rows / columns of empty slots
//                                             and the diagonal are unused
//
// Everything the TypeScript port (app/src/relhead.ts) does on the CPU runs
// in-graph: mask > 0, mask bounding boxes (coordinate ramps + ReduceMax),
// areas and pairwise intersections (BatchMatMul), the 8 pair features, the
// 2-layer transformer over the slots (empty slots masked out of attention)
// and the pair MLP. Calibration, smoothing and the motion rules stay on the
// host.

#ifndef SAM2_WEB_TENSORAPI_CC_RELHEAD_GRAPH_H_
#define SAM2_WEB_TENSORAPI_CC_RELHEAD_GRAPH_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "chain_graphs.h"

namespace sam2_chain {

inline constexpr char kRelSignature[] = "relations";
inline constexpr char kRelClsInput[] = "rel_cls";
inline constexpr char kRelValidInput[] = "rel_valid";
inline constexpr char kRelLogitsOutput[] = "rel_logits";

struct RelHeadWeights {
  int d = 0;        // token width
  int heads = 4;    // attention heads (train_head.py: 4)
  int layers = 0;   // transformer layers
  int classes = 0;  // embedding rows (PSG classes + 1 for unknown)
  int preds = 0;    // predicates
  struct Entry {
    std::vector<int> shape;
    std::vector<float> values;
  };
  absl::flat_hash_map<std::string, Entry> tensors;  // PyTorch state_dict names

  const Entry& at(const std::string& name) const;
};

// Reads relhead.safetensors (ram/export_tensorapi.py) and infers the config
// from the tensor shapes.
absl::StatusOr<RelHeadWeights> LoadRelHeadWeights(const std::string& path);

// Adds the "relations" signature for masks of side `mask_side` (S/4).
absl::Status AddRelationSignature(ModelFactory& factory,
                                  const RelHeadWeights& weights, int mask_side);

}  // namespace sam2_chain

#endif  // SAM2_WEB_TENSORAPI_CC_RELHEAD_GRAPH_H_
