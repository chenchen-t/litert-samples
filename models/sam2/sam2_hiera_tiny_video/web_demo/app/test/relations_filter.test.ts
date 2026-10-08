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

// The predicate filter (picker) on RelationEngine.

import {describe, expect, it} from 'vitest';
import {PREDICATE_GROUPS, PREDICATE_PRESETS, RelationEngine, type RelationScorer} from '../src/relations';

const side = 16;
const box = (x0: number, y0: number, x1: number, y1: number) => {
  const m = new Float32Array(side * side).fill(-1);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) m[y * side + x] = 1;
  return m;
};

/** Fixed scores for every pair. */
const fixed = (scores: Record<string, number>): RelationScorer => ({
  predicates: Object.keys(scores), symmetric: new Set(), score: () => ({...scores}),
});

describe('RelationEngine.allowed', () => {
  const objs = [{id: 1, label: 'person', mask: box(1, 1, 6, 12)}, {id: 2, label: 'ball', mask: box(8, 10, 10, 12)}];

  it('shows the best allowed predicate per pair, re-picked without re-scoring', () => {
    const e = new RelationEngine({scorer: fixed({'looking at': 0.9, kicking: 0.6, beside: 0.5}), window: 1});
    e.observe(0, 0, objs);
    expect(e.relationsAt(0).map((r) => r.predicate)).toEqual(['looking at']);
    e.allowed = new Set(['kicking', 'beside']);
    expect(e.relationsAt(0).map((r) => r.predicate)).toEqual(['kicking']);
    e.allowed = new Set(['beside']);
    expect(e.relationsAt(0).map((r) => r.predicate)).toEqual(['beside']);
    e.allowed = new Set();
    expect(e.relationsAt(0)).toEqual([]);
    e.allowed = null;
    expect(e.relationsAt(0).map((r) => r.predicate)).toEqual(['looking at']);
  });

  it('still applies the score threshold to allowed predicates', () => {
    const e = new RelationEngine({scorer: fixed({'looking at': 0.9, kicking: 0.2}), window: 1});
    e.observe(0, 0, objs);
    e.allowed = new Set(['kicking']);
    expect(e.relationsAt(0)).toEqual([]);
  });

  it('groups cover every PSG predicate once; presets only use grouped names', () => {
    const all = PREDICATE_GROUPS.flatMap(([, l]) => l);
    expect(new Set(all).size).toBe(all.length);
    for (const list of Object.values(PREDICATE_PRESETS)) for (const p of list) expect(all).toContain(p);
  });
});
