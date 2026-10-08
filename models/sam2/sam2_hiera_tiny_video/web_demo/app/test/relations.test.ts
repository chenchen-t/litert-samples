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

import {describe, expect, it} from 'vitest';

import {geometry, kindOf, pairFeatures, RelationEngine, RelationLog} from '../src/relations';

describe('RelationLog', () => {
  const rel = (predicate: string, score = 0.8) => ({s: 1, sLabel: 'player', predicate, o: 2, oLabel: 'ball', score});
  const build = () => {
    const log = new RelationLog();
    log.clear({source: 'clip', fps: 24});
    // kicking on 0-4 with a 2-frame dropout at 2-3 (bridged), chasing on 10-12, nothing on 5-9.
    for (const f of [0, 1, 4]) log.record(f, f / 24, [rel('kicking', 0.5 + f / 10)]);
    for (let f = 5; f < 10; f++) log.record(f, f / 24, []);
    for (const f of [10, 11, 12]) log.record(f, f / 24, [rel('chasing')]);
    return log;
  };
  it('keeps frame idx -> relations, sorted', () => {
    const log = build();
    expect(log.list().map((f) => f.frame)).toEqual([0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(log.get(4)?.relations[0].predicate).toBe('kicking');
    expect(log.get(7)?.relations).toEqual([]);
  });
  it('re-recording a frame replaces it', () => {
    const log = build();
    log.record(4, 4 / 24, [rel('touching')]);
    expect(log.get(4)?.relations.map((r) => r.predicate)).toEqual(['touching']);
    expect(log.size).toBe(11);
  });
  it('summarizes spans, bridging small gaps', () => {
    const spans = build().spans();
    expect(spans.map((s) => [s.predicate, s.start, s.end, s.frames])).toEqual([
      ['kicking', 0, 4, 3], ['chasing', 10, 12, 3]]);
    expect(spans[0].peak).toBeCloseTo(0.9);
  });
  it('exports JSON and CSV', () => {
    const log = build();
    const j = log.toJSON();
    expect(j.meta.source).toBe('clip');
    expect(j.frames).toHaveLength(11);
    expect(j.spans).toHaveLength(2);
    const csv = log.toCSV().trim().split('\n');
    expect(csv[0]).toBe('frame,time_s,subject_id,subject,predicate,object_id,object,score');
    expect(csv).toHaveLength(1 + 6);
    expect(csv[1]).toBe('0,0.000,1,player,kicking,2,ball,0.500');
  });
  it('caps camera-length logs, dropping the oldest frames', () => {
    const log = new RelationLog(100);
    for (let f = 0; f < 250; f++) log.record(f, f / 30, []);
    expect(log.size).toBe(100);
    expect(log.list()[0].frame).toBe(150);
  });
});

const SIDE = 96;

/** Logit mask with a filled rectangle (normalized coords). */
function rect(x0: number, y0: number, x1: number, y1: number): Float32Array {
  const m = new Float32Array(SIDE * SIDE).fill(-10);
  for (let y = Math.round(y0 * SIDE); y < Math.round(y1 * SIDE); y++) {
    for (let x = Math.round(x0 * SIDE); x < Math.round(x1 * SIDE); x++) m[y * SIDE + x] = 10;
  }
  return m;
}

/** Everything except a rectangle (e.g. a field around a player). */
function surfaceAround(x0: number, y0: number, x1: number, y1: number, top = 0.5): Float32Array {
  const m = rect(0, top, 1, 1);
  const hole = rect(x0, y0, x1, y1);
  for (let i = 0; i < m.length; i++) if (hole[i] > 0) m[i] = -10;
  return m;
}

function top(engine: RelationEngine, t: number, s: number, o: number) {
  return engine.relationsAt(t).find((r) => (r.s === s && r.o === o) || (r.s === o && r.o === s));
}

describe('kindOf', () => {
  it('maps free-form Gemma labels to kinds', () => {
    expect(kindOf('player in red')).toBe('person');
    expect(kindOf('soccer ball')).toBe('ball');
    expect(kindOf('grass field')).toBe('surface');
    expect(kindOf('bicycle')).toBe('vehicle');
    expect(kindOf('cup')).toBe('thing');
  });
});

describe('geometry', () => {
  it('computes area, centroid and box', () => {
    const g = geometry(1, 'ball', rect(0.25, 0.25, 0.75, 0.75))!;
    expect(g.area).toBeCloseTo(0.25, 2);
    expect(g.cx).toBeCloseTo(0.5, 2);
    expect(g.cy).toBeCloseTo(0.5, 2);
    expect(g.x0).toBeCloseTo(0.25, 2);
    expect(g.y1).toBeCloseTo(0.75, 2);
  });
  it('ignores empty masks', () => {
    expect(geometry(1, 'x', new Float32Array(SIDE * SIDE).fill(-1))).toBeNull();
  });
  it('detects contact between adjacent masks', () => {
    const a = geometry(1, 'a', rect(0.2, 0.2, 0.4, 0.6))!;
    const b = geometry(2, 'b', rect(0.4, 0.2, 0.6, 0.6))!;
    const far = geometry(3, 'c', rect(0.8, 0.2, 0.95, 0.6))!;
    expect(pairFeatures(a, b).contact).toBeGreaterThan(0.5);
    expect(pairFeatures(a, far).contact).toBe(0);
  });
});

describe('RelationEngine (heuristic scorer)', () => {
  it('player kicking the ball at their feet', () => {
    const e = new RelationEngine();
    for (let t = 0; t < 4; t++) {
      e.observe(t, t / 30, [
        {id: 1, label: 'player', mask: rect(0.4, 0.2, 0.5, 0.7)},
        {id: 2, label: 'soccer ball', mask: rect(0.5, 0.63, 0.54, 0.7)},
      ]);
    }
    expect(top(e, 3, 1, 2)).toMatchObject({s: 1, o: 2, predicate: 'kicking'});
  });

  it('person holding a cup', () => {
    const e = new RelationEngine();
    e.observe(0, 0, [
      {id: 1, label: 'woman', mask: rect(0.3, 0.1, 0.5, 0.9)},
      {id: 2, label: 'cup', mask: rect(0.5, 0.4, 0.56, 0.48)},
    ]);
    expect(top(e, 0, 1, 2)).toMatchObject({s: 1, o: 2, predicate: 'holding'});
  });

  it('person standing on the grass', () => {
    const e = new RelationEngine();
    e.observe(0, 0, [
      {id: 1, label: 'player', mask: rect(0.45, 0.3, 0.55, 0.8)},
      {id: 2, label: 'grass', mask: surfaceAround(0.45, 0.3, 0.55, 0.8, 0.6)},
    ]);
    expect(top(e, 0, 1, 2)).toMatchObject({s: 1, o: 2, predicate: 'standing on'});
  });

  it('cup in a box (containment)', () => {
    const e = new RelationEngine();
    e.observe(0, 0, [
      {id: 1, label: 'cup', mask: rect(0.45, 0.45, 0.5, 0.5)},
      {id: 2, label: 'box', mask: rect(0.3, 0.3, 0.7, 0.7)},
    ]);
    expect(top(e, 0, 1, 2)).toMatchObject({s: 1, o: 2, predicate: 'in'});
  });

  it('player chasing another player (same direction, behind, moving)', () => {
    const e = new RelationEngine();
    for (let t = 0; t < 8; t++) {
      const dx = t * 0.02;  // 0.6 frame widths / s at 30 fps
      e.observe(t, t / 30, [
        {id: 1, label: 'player', mask: rect(0.1 + dx, 0.3, 0.16 + dx, 0.6)},
        {id: 2, label: 'player', mask: rect(0.35 + dx * 0.9, 0.3, 0.41 + dx * 0.9, 0.6)},
      ]);
    }
    expect(top(e, 7, 1, 2)).toMatchObject({s: 1, o: 2, predicate: 'chasing'});
  });

  it('static, separated objects side by side are "beside" once per pair', () => {
    const e = new RelationEngine();
    e.observe(0, 0, [
      {id: 1, label: 'chair', mask: rect(0.3, 0.4, 0.4, 0.6)},
      {id: 2, label: 'lamp', mask: rect(0.45, 0.4, 0.55, 0.6)},
    ]);
    const rels = e.relationsAt(0);
    expect(rels).toHaveLength(1);
    expect(rels[0].predicate).toBe('beside');
  });

  it('far apart objects have no relation', () => {
    const e = new RelationEngine();
    e.observe(0, 0, [
      {id: 1, label: 'chair', mask: rect(0.02, 0.02, 0.08, 0.08)},
      {id: 2, label: 'lamp', mask: rect(0.9, 0.9, 0.96, 0.96)},
    ]);
    expect(e.relationsAt(0)).toHaveLength(0);
  });

  it('smooths over the window: a one-frame glitch does not flip the relation', () => {
    const e = new RelationEngine();
    for (let t = 0; t < 6; t++) {
      const glitch = t === 5;
      e.observe(t, t / 30, [
        {id: 1, label: 'woman', mask: rect(0.3, 0.1, 0.5, 0.9)},
        {id: 2, label: 'cup', mask: glitch ? rect(0.8, 0.4, 0.86, 0.48) : rect(0.5, 0.4, 0.56, 0.48)},
      ]);
    }
    expect(top(e, 5, 1, 2)?.predicate).toBe('holding');
  });

  it('6 objects score in well under a frame', () => {
    const e = new RelationEngine();
    const objs = Array.from({length: 6}, (_, k) => ({id: k + 1, label: k ? 'player' : 'ball',
      mask: rect(0.1 + k * 0.13, 0.3, 0.2 + k * 0.13, 0.7)}));
    let ms = 0;
    for (let t = 0; t < 10; t++) ms = Math.max(ms, e.observe(t, t / 30, objs));
    expect(ms).toBeLessThan(15);
  });
});
