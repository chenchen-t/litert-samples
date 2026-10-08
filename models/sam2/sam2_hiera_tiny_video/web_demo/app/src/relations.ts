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

// Real-time relations between tracked objects ("Relate Anything", phase 1).
//
// Every tracked frame gives one low-res SAM 2 mask per object (logits,
// [S/4, S/4], model space = the frame squashed to a square, so normalized
// coordinates map straight to the frame). From these masks, the Gemma labels
// and the objects' motion over a short window, a scorer rates a fixed set of
// predicates for every ordered pair. Scores are averaged over the window
// (temporal smoothing, made possible by SAM 2's stable object ids) and the
// best predicate per pair is shown.
//
// The scorer is a plug-in (RelationScorer): this file ships a geometry +
// label + motion scorer that needs no training. A learned head over SAM 2's
// object pointers (see ram_research.md, option B) can replace it with the
// same inputs plus features.

/** One tracked object on one frame. */
export interface RelObject {
  id: number;
  label: string;
  /** PSG class of the label ('' or absent = unknown), for learned scorers. */
  cls?: string;
  mask: Float32Array;  // low-res logits, side x side
}

export interface Relation {
  s: number;  // subject object id
  o: number;  // object id
  predicate: string;
  score: number;  // smoothed, 0..1
}

/** Per-object mask geometry, normalized [0,1] frame coordinates. */
export interface Geom {
  id: number;
  kind: Kind;
  cls: string;  // PSG class or ''
  side: number;
  bits: Uint8Array;     // mask > 0
  dilated: Uint8Array;  // bits dilated by DILATE px
  count: number;
  area: number;  // fraction of the frame
  cx: number;
  cy: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  bottom: number;  // pixels in the lowest BAND of the mask's rows
}

/** Pair features of (s, o) on one frame plus the window's motion. */
export interface PairFeatures {
  s: Geom;
  o: Geom;
  interS: number;   // |s ∩ o| / |s|
  interO: number;   // |s ∩ o| / |o|
  contact: number;  // 0..1, dilated boundary overlap
  bottomOn: number; // fraction of s's bottom band touching o
  dist: number;     // centroid distance / mean object size
  dx: number;       // o.cx - s.cx (normalized frame units)
  dy: number;
  hOverlap: number; // horizontal bbox overlap / narrower width
  // Motion (normalized frame units per second; 0 when unknown).
  vs: [number, number];
  vo: [number, number];
  closing: number;  // -d(dist)/dt in frame units per second (> 0: getting closer)
}

export type Kind = 'person' | 'animal' | 'ball' | 'surface' | 'vehicle' | 'thing';

const KIND_WORDS: Array<[Kind, RegExp]> = [
  ['person', /\b(person|people|man|men|woman|women|boy|girl|child|children|kid|player|players|referee|goalkeeper|keeper|athlete|human|guy|lady|dancer|runner|rider|skater|surfer|someone|he|she)\b/i],
  ['ball', /\b(ball|football|soccer ?ball|basketball|volleyball|tennis ball|baseball|frisbee)\b/i],
  ['animal', /\b(dog|cat|horse|bird|cow|sheep|animal|puppy|kitten|pet)\b/i],
  ['surface', /\b(grass|field|playingfield|pitch|floor|ground|road|street|pavement|table|desk|court|sand|beach|snow|water|bed|sofa|couch|bench|chair|carpet|rug|stage|track|lawn|turf|dirt|gravel|platform)\b/i],
  ['vehicle', /\b(car|bike|bicycle|motorcycle|bus|truck|boat|skateboard|surfboard|train|scooter)\b/i],
];

export function kindOf(label: string): Kind {
  for (const [k, re] of KIND_WORDS) if (re.test(label)) return k;
  return 'thing';
}

const DILATE = 2;     // px at S/4 (96 px at 384): about 2% of the frame
const BAND = 0.12;    // bottom band height, fraction of the mask's height

export function geometry(id: number, label: string, mask: Float32Array, cls = ''): Geom | null {
  const side = Math.round(Math.sqrt(mask.length));
  if (side * side !== mask.length) return null;
  const bits = new Uint8Array(mask.length);
  let count = 0, sx = 0, sy = 0, x0 = side, y0 = side, x1 = -1, y1 = -1;
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      if (mask[y * side + x] > 0) {
        bits[y * side + x] = 1;
        count++;
        sx += x;
        sy += y;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (count < 3) return null;
  const dilated = dilate(bits, side, DILATE);
  const bandTop = y1 - Math.max(1, Math.round((y1 - y0 + 1) * BAND)) + 1;
  let bottom = 0;
  for (let y = bandTop; y <= y1; y++) for (let x = x0; x <= x1; x++) bottom += bits[y * side + x];
  const n = side;
  const kind = kindOf(label);
  return {
    id, kind: kind === 'thing' && cls ? kindOf(cls.replace(/-/g, ' ')) : kind, cls, side, bits, dilated, count,
    area: count / (n * n),
    cx: (sx / count + 0.5) / n, cy: (sy / count + 0.5) / n,
    x0: x0 / n, y0: y0 / n, x1: (x1 + 1) / n, y1: (y1 + 1) / n, bottom,
  };
}

/** Square dilation (separable max filter). */
function dilate(bits: Uint8Array, side: number, r: number): Uint8Array {
  const tmp = new Uint8Array(bits.length), out = new Uint8Array(bits.length);
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      let v = 0;
      for (let k = Math.max(0, x - r); k <= Math.min(side - 1, x + r) && !v; k++) v = bits[y * side + k];
      tmp[y * side + x] = v;
    }
  }
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      let v = 0;
      for (let k = Math.max(0, y - r); k <= Math.min(side - 1, y + r) && !v; k++) v = tmp[k * side + x];
      out[y * side + x] = v;
    }
  }
  return out;
}

export function pairFeatures(s: Geom, o: Geom): PairFeatures {
  const n = s.bits.length;
  let inter = 0, touch = 0;
  for (let i = 0; i < n; i++) {
    if (s.bits[i] && o.bits[i]) inter++;
    // Boundary contact: s's dilated rim meets o (or o's rim meets s).
    if ((s.dilated[i] && o.bits[i] && !s.bits[i]) || (o.dilated[i] && s.bits[i] && !o.bits[i])) touch++;
  }
  // Bottom band of s against o (dilated): "standing on", "lying on".
  const side = s.side;
  const sy1 = Math.round(s.y1 * side) - 1, sy0 = Math.round(s.y0 * side);
  const bandTop = sy1 - Math.max(1, Math.round((sy1 - sy0 + 1) * BAND)) + 1;
  const sx0 = Math.round(s.x0 * side), sx1 = Math.round(s.x1 * side) - 1;
  let bottomHit = 0;
  for (let y = bandTop; y <= sy1; y++) {
    for (let x = sx0; x <= sx1; x++) {
      const i = y * side + x;
      // Also look a few rows below the band: supports sit just under the mask.
      if (s.bits[i] && (o.dilated[i] || (y + DILATE < side && o.bits[i + DILATE * side]))) bottomHit++;
    }
  }
  const size = (Math.sqrt(s.area) + Math.sqrt(o.area)) / 2;
  const dx = o.cx - s.cx, dy = o.cy - s.cy;
  const wS = s.x1 - s.x0, wO = o.x1 - o.x0;
  const hOverlap = Math.max(0, Math.min(s.x1, o.x1) - Math.max(s.x0, o.x0)) / Math.max(1e-6, Math.min(wS, wO));
  // Contact strength: touching pixels relative to the smaller object's rim length.
  const rim = 2 * Math.sqrt(Math.min(s.count, o.count));
  return {
    s, o,
    interS: inter / s.count, interO: inter / o.count,
    contact: Math.min(1, touch / Math.max(1, rim)),
    bottomOn: s.bottom ? bottomHit / s.bottom : 0,
    dist: Math.hypot(dx, dy) / Math.max(1e-6, size),
    dx, dy, hOverlap,
    vs: [0, 0], vo: [0, 0], closing: 0,
  };
}

/** Ramps 0 -> 1 between a and b. */
const ramp = (x: number, a: number, b: number) => Math.min(1, Math.max(0, (x - a) / (b - a)));

/** Rates predicates for one ordered pair on one frame. */
export interface RelationScorer {
  readonly predicates: readonly string[];
  /** Symmetric predicates: shown once per unordered pair. */
  readonly symmetric: ReadonlySet<string>;
  score(f: PairFeatures): Record<string, number>;
  /**
   * Optional: scores all pairs of a frame at once (scorers that look at the
   * whole scene, e.g. the learned head). `pairs` is keyed "s,o"; returns the
   * same keys.
   */
  scoreFrame?(geoms: Geom[], pairs: Map<string, PairFeatures>): Map<string, Record<string, number>>;
}

/**
 * Geometry + label + motion rules (no training). Predicate names follow the
 * PSG vocabulary where one exists, so a PSG-trained head can drop in.
 */
export class HeuristicScorer implements RelationScorer {
  readonly predicates = ['kicking', 'holding', 'riding', 'standing on', 'on', 'in', 'touching',
    'chasing', 'approaching', 'moving away', 'beside', 'near', 'above'] as const;
  readonly symmetric = new Set(['touching', 'beside', 'near']);

  score(f: PairFeatures): Record<string, number> {
    const {s, o} = f;
    const r: Record<string, number> = {};
    const contact = Math.max(ramp(f.contact, 0.05, 0.35), ramp(f.interO, 0.02, 0.2) * ramp(f.interS, 0.02, 0.2));
    const near = 1 - ramp(f.dist, 1.2, 2.5);
    const smaller = ramp(s.area / Math.max(1e-6, o.area), 1.2, 3);  // s bigger than o
    // o's centroid relative to s's box (0 = top, 1 = bottom).
    const relY = (o.cy - s.y0) / Math.max(1e-6, s.y1 - s.y0);
    // o's box reaches s's box horizontally (held / kicked things stick out of it).
    const margin = 0.03;
    const insideBoxX = o.x0 <= s.x1 + margin && o.x1 >= s.x0 - margin ? 1 : 0;

    if (s.kind === 'person' && o.kind === 'ball') {
      // Ball at the feet and in contact (or about to be).
      r.kicking = Math.max(contact, 0.6 * (1 - ramp(f.dist, 0.6, 1.2))) * ramp(relY, 0.6, 0.85) * insideBoxX;
    }
    if ((s.kind === 'person' || s.kind === 'animal') && o.kind !== 'surface' && o.kind !== 'person' &&
        o.kind !== 'vehicle') {
      r.holding = contact * smaller * ramp(0.85 - relY, 0, 0.25) * insideBoxX;
    }
    if (s.kind === 'person' && (o.kind === 'vehicle' || o.kind === 'animal')) {
      // Rider above the vehicle / animal, in contact, horizontally aligned.
      r.riding = contact * ramp(o.cy - s.cy, 0.02, 0.1) * ramp(f.hOverlap, 0.3, 0.7);
    }
    // Support: s's bottom rests on o, s mostly above o.
    const support = ramp(f.bottomOn, 0.2, 0.6) * ramp(f.dy, -0.02, 0.05) * (1 - ramp(f.interS, 0.6, 0.9));
    if (o.kind === 'surface' || o.area > 1.5 * s.area) {
      if (s.kind === 'person' || s.kind === 'animal') r['standing on'] = support;
      else r.on = support;
    }
    // Containment: s lies (almost) entirely inside o's region.
    r.in = ramp(f.interS, 0.6, 0.9) * ramp(o.area / Math.max(1e-6, s.area), 1.5, 3);

    r.touching = contact * 0.75;

    // Motion predicates (window velocities).
    const speed = (v: [number, number]) => Math.hypot(v[0], v[1]);
    const sp = speed(f.vs), op = speed(f.vo);
    const moving = ramp(sp, 0.03, 0.12);  // frame units / s
    if (sp > 1e-6) {
      const toward = (f.vs[0] * f.dx + f.vs[1] * f.dy) / (sp * Math.max(1e-6, Math.hypot(f.dx, f.dy)));
      r.approaching = moving * ramp(toward, 0.3, 0.8) * ramp(f.closing, 0.02, 0.1) * (1 - contact) * 0.8;
      r['moving away'] = moving * ramp(-toward, 0.3, 0.8) * ramp(-f.closing, 0.02, 0.1) * near * 0.7;
      const animate = s.kind === 'person' || s.kind === 'animal' || s.kind === 'vehicle';
      if (op > 1e-6 && animate && o.kind !== 'surface') {
        const same = (f.vs[0] * f.vo[0] + f.vs[1] * f.vo[1]) / (sp * op);
        r.chasing = moving * ramp(op, 0.03, 0.12) * ramp(same, 0.5, 0.85) * ramp(toward, 0.2, 0.7) *
            (1 - ramp(f.dist, 3, 6)) * 0.9;
      }
    }

    // Static layout, weakest: only when nothing stronger applies.
    r.beside = near * ramp(1 - Math.abs(f.dy) / Math.max(1e-6, Math.abs(f.dx) + 1e-3), 0.3, 0.7) *
        (1 - contact) * 0.6;
    r.near = near * (1 - contact) * 0.5;
    r.above = ramp(f.hOverlap, 0.3, 0.7) * ramp(-f.dy, 0.05, 0.15) * (1 - contact) * near * 0.5;
    return r;
  }
}

interface FrameObs {
  t: number;
  time: number;  // seconds
  geoms: Map<number, Geom>;
  scores: Map<string, Record<string, number>>;  // "s,o" -> predicate scores
}

export interface RelationOptions {
  window?: number;     // frames averaged
  minScore?: number;   // show threshold on the smoothed score
  maxEdges?: number;   // per frame
  scorer?: RelationScorer;
}

/** Predicate groups for the picker (PSG predicates + the motion / rule ones). */
export const PREDICATE_GROUPS: ReadonlyArray<[string, readonly string[]]> = [
  ['Position', ['over', 'above', 'in front of', 'beside', 'near', 'on', 'in', 'on back of', 'attached to',
    'hanging from', 'leaning on', 'enclosing', 'painted on', 'parked on']],
  ['On a surface', ['standing on', 'sitting on', 'lying on', 'walking on', 'running on', 'driving on']],
  ['Moving through', ['crossing', 'entering', 'exiting', 'going down', 'jumping over', 'jumping from', 'flying over']],
  ['Hands / body', ['holding', 'carrying', 'wearing', 'touching', 'pushing', 'pulling', 'throwing', 'catching',
    'kicking', 'swinging', 'about to hit']],
  ['Activities', ['playing', 'playing with', 'riding', 'driving', 'guiding', 'feeding', 'eating', 'drinking',
    'biting', 'cooking', 'slicing']],
  ['Attention / social', ['looking at', 'talking to', 'kissing']],
  ['Motion', ['approaching', 'moving away', 'chasing']],
];

/** Preset predicate sets for the picker. */
export const PREDICATE_PRESETS: Record<string, readonly string[]> = {
  Sports: ['kicking', 'throwing', 'catching', 'swinging', 'about to hit', 'playing', 'playing with', 'holding',
    'looking at', 'standing on', 'running on', 'walking on', 'jumping over', 'touching', 'pushing', 'pulling',
    'approaching', 'moving away', 'chasing'],
  Actions: ['holding', 'carrying', 'pushing', 'pulling', 'throwing', 'catching', 'kicking', 'swinging', 'about to hit',
    'playing', 'playing with', 'riding', 'driving', 'guiding', 'feeding', 'eating', 'drinking', 'biting', 'cooking',
    'slicing', 'looking at', 'talking to', 'kissing', 'approaching', 'moving away', 'chasing'],
  Spatial: ['over', 'above', 'in front of', 'beside', 'near', 'on', 'in', 'on back of', 'attached to', 'hanging from',
    'leaning on', 'enclosing', 'standing on', 'sitting on', 'lying on'],
  Motion: ['approaching', 'moving away', 'chasing', 'walking on', 'running on', 'crossing', 'entering', 'exiting',
    'going down', 'jumping over', 'jumping from', 'flying over'],
};

export class RelationEngine {
  private _scorer: RelationScorer;
  private readonly frames = new Map<number, FrameObs>();
  private readonly window: number;
  private readonly minScore: number;
  private readonly maxEdges: number;
  /**
   * Predicates that may be shown (null = all). Applied when each pair's best
   * predicate is picked, so a pair shows its best *allowed* predicate; scores
   * are kept for all, so changing it needs no re-scoring.
   */
  allowed: ReadonlySet<string> | null = null;

  constructor(o: RelationOptions = {}) {
    this._scorer = o.scorer ?? new HeuristicScorer();
    this.window = o.window ?? 6;
    this.minScore = o.minScore ?? 0.45;
    this.maxEdges = o.maxEdges ?? 8;
  }

  get scorer(): RelationScorer { return this._scorer; }
  /** Switches the scorer; recorded frames are dropped (their scores came from the old one). */
  set scorer(s: RelationScorer) {
    this._scorer = s;
    this.frames.clear();
  }

  reset() { this.frames.clear(); }
  /** Drops frames older than `keep` frames behind t (live camera). */
  prune(t: number, keep = 30) {
    for (const k of this.frames.keys()) if (k < t - keep || k > t) this.frames.delete(k);
  }
  has(t: number) { return this.frames.has(t); }

  /** Records frame t (time in seconds) and scores its pairs. Returns ms spent. */
  observe(t: number, time: number, objs: RelObject[]): number {
    const t0 = performance.now();
    const geoms = new Map<number, Geom>();
    for (const ob of objs) {
      const g = geometry(ob.id, ob.label, ob.mask, ob.cls ?? '');
      if (g) geoms.set(ob.id, g);
    }
    const prev = this.previous(t);
    const vel = (id: number): [number, number] => {
      const a = prev?.geoms.get(id), b = geoms.get(id);
      if (!prev || !a || !b) return [0, 0];
      const dt = Math.max(1e-3, time - prev.time);
      return [(b.cx - a.cx) / dt, (b.cy - a.cy) / dt];
    };
    const pairs = new Map<string, PairFeatures>();
    for (const s of geoms.values()) {
      for (const o of geoms.values()) {
        if (s.id === o.id) continue;
        const f = pairFeatures(s, o);
        f.vs = vel(s.id);
        f.vo = vel(o.id);
        const ps = prev?.geoms.get(s.id), po = prev?.geoms.get(o.id);
        if (prev && ps && po) {
          const dt = Math.max(1e-3, time - prev.time);
          f.closing = (Math.hypot(po.cx - ps.cx, po.cy - ps.cy) - Math.hypot(f.dx, f.dy)) / dt;
        }
        pairs.set(`${s.id},${o.id}`, f);
      }
    }
    const sc = this._scorer;
    const scores = sc.scoreFrame ? sc.scoreFrame([...geoms.values()], pairs)
      : new Map([...pairs].map(([k, f]) => [k, sc.score(f)]));
    this.frames.set(t, {t, time, geoms, scores});
    return performance.now() - t0;
  }

  /** A frame ~3 frames back for velocities (less jitter than t-1). */
  private previous(t: number): FrameObs | undefined {
    for (let k = 3; k >= 1; k--) {
      const f = this.frames.get(t - k);
      if (f) return f;
    }
    return undefined;
  }

  /** Centroid of object `id` on frame t (normalized), for drawing. */
  centroid(t: number, id: number): [number, number] | null {
    const g = this.frames.get(t)?.geoms.get(id);
    return g ? [g.cx, g.cy] : null;
  }

  /** Smoothed relations on frame t among `ids` (defaults to all on that frame). */
  relationsAt(t: number, ids?: ReadonlySet<number>): Relation[] {
    const cur = this.frames.get(t);
    if (!cur) return [];
    const win: FrameObs[] = [];
    for (let k = 0; k < this.window * 2 && win.length < this.window; k++) {
      const f = this.frames.get(t - k);
      if (f) win.push(f);
    }
    const best: Relation[] = [];
    for (const key of cur.scores.keys()) {
      const [s, o] = key.split(',').map(Number);
      if (ids && (!ids.has(s) || !ids.has(o))) continue;
      const avg: Record<string, number> = {};
      let n = 0;
      for (const f of win) {
        const sc = f.scores.get(key);
        if (!sc) continue;
        n++;
        for (const [p, v] of Object.entries(sc)) avg[p] = (avg[p] ?? 0) + v;
      }
      let top: Relation | null = null;
      for (const [p, v] of Object.entries(avg)) {
        if (this.allowed && !this.allowed.has(p)) continue;
        const m = v / Math.max(1, n);
        if (!top || m > top.score) top = {s, o, predicate: p, score: m};
      }
      if (top && top.score >= this.minScore) best.push(top);
    }
    // Symmetric predicates once per unordered pair; at most one edge per pair.
    best.sort((a, b) => b.score - a.score);
    const seen = new Set<string>();
    const out: Relation[] = [];
    for (const r of best) {
      const pair = r.s < r.o ? `${r.s},${r.o}` : `${r.o},${r.s}`;
      if (seen.has(pair)) continue;
      seen.add(pair);
      out.push(r);
      if (out.length >= this.maxEdges) break;
    }
    return out;
  }
}

// ---------------------------------------------------------------- log

/** One relation as logged (labels resolved when it was recorded). */
export interface LoggedRelation {
  s: number;
  sLabel: string;
  predicate: string;
  o: number;
  oLabel: string;
  score: number;
}

/** Frame idx -> its relations. `time` is seconds into the video / camera session. */
export interface LogFrame {
  frame: number;
  time: number;
  relations: LoggedRelation[];
}

/** A relation held over consecutive frames (small gaps bridged). */
export interface LogSpan {
  s: number;
  sLabel: string;
  predicate: string;
  o: number;
  oLabel: string;
  start: number;  // first frame
  end: number;    // last frame
  startTime: number;
  endTime: number;
  frames: number;  // frames it was shown on
  peak: number;    // max score
}

export interface LogMeta {
  source: string;  // video name or "camera"
  fps?: number;
  width?: number;
  height?: number;
  startedAt: string;  // ISO time the log started
}

/**
 * The relation log: every scored frame's relations, kept so they can be traced
 * back (jump to a frame), summarized as spans, and saved as JSON / CSV.
 * Re-recording a frame replaces it (re-tracking).
 */
export class RelationLog {
  private readonly frames = new Map<number, LogFrame>();
  meta: LogMeta = {source: '', startedAt: new Date().toISOString()};

  constructor(private readonly maxFrames = 20000, private readonly gap = 2) {}

  clear(meta?: Partial<LogMeta>) {
    this.frames.clear();
    this.meta = {source: '', ...meta, startedAt: new Date().toISOString()};
  }

  get size() { return this.frames.size; }

  record(frame: number, time: number, relations: LoggedRelation[]) {
    this.frames.delete(frame);  // keep insertion order = recording order
    this.frames.set(frame, {frame, time, relations});
    while (this.frames.size > this.maxFrames) {
      this.frames.delete(this.frames.keys().next().value!);
    }
  }

  get(frame: number): LogFrame | undefined { return this.frames.get(frame); }

  /** All logged frames by frame idx. */
  list(): LogFrame[] { return [...this.frames.values()].sort((a, b) => a.frame - b.frame); }

  /** Relations as spans over frames, by start frame. */
  spans(): LogSpan[] {
    const open = new Map<string, LogSpan>();
    const done: LogSpan[] = [];
    for (const f of this.list()) {
      for (const r of f.relations) {
        const key = `${r.s}|${r.predicate}|${r.o}`;
        const sp = open.get(key);
        if (sp && f.frame - sp.end <= this.gap + 1) {
          sp.end = f.frame;
          sp.endTime = f.time;
          sp.frames++;
          sp.peak = Math.max(sp.peak, r.score);
        } else {
          if (sp) done.push(sp);
          open.set(key, {s: r.s, sLabel: r.sLabel, predicate: r.predicate, o: r.o, oLabel: r.oLabel,
            start: f.frame, end: f.frame, startTime: f.time, endTime: f.time, frames: 1, peak: r.score});
        }
      }
    }
    done.push(...open.values());
    return done.sort((a, b) => a.start - b.start || a.end - b.end);
  }

  toJSON() {
    return {meta: this.meta, frames: this.list(), spans: this.spans()};
  }

  /** One row per (frame, relation). */
  toCSV(): string {
    const q = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
    const rows = ['frame,time_s,subject_id,subject,predicate,object_id,object,score'];
    for (const f of this.list()) {
      for (const r of f.relations) {
        rows.push([f.frame, f.time.toFixed(3), r.s, q(r.sLabel), q(r.predicate), r.o, q(r.oLabel),
          r.score.toFixed(3)].join(','));
      }
    }
    return rows.join('\n') + '\n';
  }
}
