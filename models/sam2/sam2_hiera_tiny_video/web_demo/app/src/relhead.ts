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

// The learned relation head (phase 2), in plain TypeScript.
//
// A PSG-trained "geom_label" head (ram/train_head.py, exported by
// ram/export_web.py): per object its mask box + area and its PSG class, a
// 2-layer transformer over the frame's objects, then an MLP per ordered pair
// on [token_s, token_o, pair geometry] -> one logit per PSG predicate.
// Inputs are the SAM 2 masks the page already reads every frame, so no extra
// model run is needed; ~0.33 M params, ~1-2 ms on the CPU for 6 objects.

import {type Geom, HeuristicScorer, type PairFeatures, type RelationScorer} from './relations';

export interface HeadMeta {
  variant: 'geom_label';
  d: number;
  heads: number;
  layers: number;
  ff: number;
  classes: string[];
  preds: string[];
  size: number;
  tensors: Record<string, {offset: number; shape: number[]}>;
  params: number;
  metrics?: Record<string, number>;
  /** Logit offset by object count (2..6; more use 6): -(F1-optimal threshold), ram/export_web.py. */
  offsets?: Record<string, number>;
}

/** One object: its binary mask (side x side, 1 = inside) and PSG class ('' = unknown). */
export interface HeadObject {
  bits: Uint8Array;
  side: number;
  cls: string;
}

interface Lin { w: Float32Array; b: Float32Array; out: number; inp: number }
interface Norm { g: Float32Array; b: Float32Array }
interface Layer { inW: Lin; outW: Lin; l1: Lin; l2: Lin; n1: Norm; n2: Norm }

const SQRT1_2 = Math.SQRT1_2;

/** erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). */
function erf(x: number): number {
  const s = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
      Math.exp(-x * x);
  return s * y;
}

const gelu = (x: number) => 0.5 * x * (1 + erf(x * SQRT1_2));

/** y[out] = W x + b (W row-major [out, inp]), x read from `x` at `xo`. */
function linear(l: Lin, x: Float32Array, xo: number, y: Float32Array, yo: number) {
  const {w, b, out, inp} = l;
  for (let i = 0; i < out; i++) {
    let s = b[i];
    const r = i * inp;
    for (let k = 0; k < inp; k++) s += w[r + k] * x[xo + k];
    y[yo + i] = s;
  }
}

function layerNorm(n: Norm, x: Float32Array, xo: number, y: Float32Array, yo: number, d: number) {
  let m = 0;
  for (let k = 0; k < d; k++) m += x[xo + k];
  m /= d;
  let v = 0;
  for (let k = 0; k < d; k++) v += (x[xo + k] - m) ** 2;
  const r = 1 / Math.sqrt(v / d + 1e-5);
  for (let k = 0; k < d; k++) y[yo + k] = (x[xo + k] - m) * r * n.g[k] + n.b[k];
}

/** Box (normalized, mask bounding box), area fraction and pixel count of a binary mask. */
export function maskBox(bits: Uint8Array, side: number) {
  let x0 = side, y0 = side, x1 = -1, y1 = -1, count = 0;
  for (let y = 0; y < side; y++) {
    for (let x = 0; x < side; x++) {
      if (!bits[y * side + x]) continue;
      count++;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  if (!count) return null;
  return {x0: x0 / side, y0: y0 / side, x1: (x1 + 1) / side, y1: (y1 + 1) / side, area: count / (side * side), count};
}

export class RelHead {
  readonly preds: readonly string[];
  readonly classes: readonly string[];
  private readonly cidx: Map<string, number>;
  private readonly d: number;
  private readonly P: number;
  private readonly emb: Float32Array;
  private readonly embDim: number;
  private readonly inp: Lin;
  private readonly inpNorm: Norm;
  private readonly layers: Layer[];
  private readonly pg: Lin;
  private readonly h0: Lin;
  private readonly h1: Lin;

  constructor(readonly meta: HeadMeta, data: Float32Array) {
    if (meta.variant !== 'geom_label') throw new Error(`relation head: unsupported variant ${meta.variant}`);
    const t = (name: string) => {
      const e = meta.tensors[name];
      if (!e) throw new Error(`relation head: missing tensor ${name}`);
      const n = e.shape.reduce((a, b) => a * b, 1);
      return {a: data.subarray(e.offset, e.offset + n), shape: e.shape};
    };
    const lin = (p: string): Lin => {
      const w = t(`${p}.weight`);
      return {w: w.a, b: t(`${p}.bias`).a, out: w.shape[0], inp: w.shape[1]};
    };
    const norm = (p: string): Norm => ({g: t(`${p}.weight`).a, b: t(`${p}.bias`).a});
    this.preds = meta.preds;
    this.classes = meta.classes;
    this.cidx = new Map(meta.classes.map((c, i) => [c, i + 1]));  // 0 = unknown
    this.d = meta.d;
    this.P = meta.preds.length;
    const e = t('emb.weight');
    this.emb = e.a;
    this.embDim = e.shape[1];
    this.inp = lin('inp.0');
    this.inpNorm = norm('inp.1');
    this.layers = [];
    for (let i = 0; i < meta.layers; i++) {
      const p = `enc.layers.${i}`;
      const inW = t(`${p}.self_attn.in_proj_weight`);
      this.layers.push({
        inW: {w: inW.a, b: t(`${p}.self_attn.in_proj_bias`).a, out: inW.shape[0], inp: inW.shape[1]},
        outW: lin(`${p}.self_attn.out_proj`), l1: lin(`${p}.linear1`), l2: lin(`${p}.linear2`),
        n1: norm(`${p}.norm1`), n2: norm(`${p}.norm2`),
      });
    }
    this.pg = lin('pg.0');
    this.h0 = lin('head.0');
    this.h1 = lin('head.3');
  }

  /** Index of a PSG class (0 = unknown / not in the head's vocabulary). */
  classIndex(cls: string): number { return this.cidx.get(cls) ?? 0; }

  /**
   * Logits [N, N, P] (row-major s, o, predicate) for N objects; the diagonal
   * is left at -Infinity. A logit > 0 means "predicted" (multi-label loss).
   */
  forward(objs: HeadObject[]): Float32Array {
    const N = objs.length, d = this.d, P = this.P;
    const out = new Float32Array(N * N * P).fill(-Infinity);
    if (N < 2) return out;
    const boxes = objs.map((o) => maskBox(o.bits, o.side) ?? {x0: 0, y0: 0, x1: 0, y1: 0, area: 0, count: 0});

    // Object tokens: [geom 9, class embedding] -> Linear -> LayerNorm -> GELU.
    const din = this.inp.inp;
    const feat = new Float32Array(N * din);
    const x = new Float32Array(N * d);
    for (let i = 0; i < N; i++) {
      const b = boxes[i], f = i * din;
      feat.set([b.x0, b.y0, b.x1, b.y1, (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, b.x1 - b.x0, b.y1 - b.y0, b.area], f);
      const c = this.classIndex(objs[i].cls);
      feat.set(this.emb.subarray(c * this.embDim, (c + 1) * this.embDim), f + 9);
      linear(this.inp, feat, f, x, i * d);
      layerNorm(this.inpNorm, x, i * d, x, i * d, d);
      for (let k = 0; k < d; k++) x[i * d + k] = gelu(x[i * d + k]);
    }

    // Transformer encoder layers (pre-norm, ReLU feed-forward).
    const H = this.meta.heads, hd = d / H, scale = 1 / Math.sqrt(hd);
    const nx = new Float32Array(N * d), qkv = new Float32Array(N * 3 * d), att = new Float32Array(N * d);
    const tmp = new Float32Array(N * d), w = new Float32Array(N);
    for (const L of this.layers) {
      for (let i = 0; i < N; i++) {
        layerNorm(L.n1, x, i * d, nx, i * d, d);
        linear(L.inW, nx, i * d, qkv, i * 3 * d);
      }
      for (let h = 0; h < H; h++) {
        for (let i = 0; i < N; i++) {
          let mx = -Infinity;
          for (let j = 0; j < N; j++) {
            let s = 0;
            for (let k = 0; k < hd; k++) s += qkv[i * 3 * d + h * hd + k] * qkv[j * 3 * d + d + h * hd + k];
            w[j] = s * scale;
            if (w[j] > mx) mx = w[j];
          }
          let z = 0;
          for (let j = 0; j < N; j++) z += (w[j] = Math.exp(w[j] - mx));
          for (let k = 0; k < hd; k++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += w[j] * qkv[j * 3 * d + 2 * d + h * hd + k];
            att[i * d + h * hd + k] = s / z;
          }
        }
      }
      for (let i = 0; i < N; i++) {
        linear(L.outW, att, i * d, tmp, i * d);
        for (let k = 0; k < d; k++) x[i * d + k] += tmp[i * d + k];
      }
      const hid = new Float32Array(L.l1.out);
      for (let i = 0; i < N; i++) {
        layerNorm(L.n2, x, i * d, nx, i * d, d);
        linear(L.l1, nx, i * d, hid, 0);
        for (let k = 0; k < hid.length; k++) hid[k] = Math.max(0, hid[k]);
        linear(L.l2, hid, 0, tmp, i * d);
        for (let k = 0; k < d; k++) x[i * d + k] += tmp[i * d + k];
      }
    }

    // Pair head. head.0 on [tok_s, tok_o, pg] = W_s tok_s + W_o tok_o + W_g pg + b:
    // the token parts are computed once per object.
    const h0 = this.h0, D = h0.out, IN = h0.inp;
    const A = new Float32Array(N * D), B = new Float32Array(N * D);
    for (let i = 0; i < N; i++) {
      for (let r = 0; r < D; r++) {
        let sa = 0, sb = 0;
        const row = r * IN;
        for (let k = 0; k < d; k++) {
          const v = x[i * d + k];
          sa += h0.w[row + k] * v;
          sb += h0.w[row + d + k] * v;
        }
        A[i * D + r] = sa;
        B[i * D + r] = sb;
      }
    }
    // Mask intersections for mask IoU / containment.
    const inter = new Float32Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const a = objs[i].bits, b = objs[j].bits;
        let c = 0;
        for (let k = 0; k < a.length; k++) c += a[k] & b[k];
        inter[i * N + j] = inter[j * N + i] = c;
      }
    }
    const geo = new Float32Array(8), g64 = new Float32Array(this.pg.out), hid = new Float32Array(D);
    const logit = new Float32Array(P);
    for (let s = 0; s < N; s++) {
      for (let o = 0; o < N; o++) {
        if (s === o) continue;
        pairGeom(boxes[s], boxes[o], inter[s * N + o], geo);
        linear(this.pg, geo, 0, g64, 0);
        for (let k = 0; k < g64.length; k++) g64[k] = gelu(g64[k]);
        for (let r = 0; r < D; r++) {
          let v = h0.b[r] + A[s * D + r] + B[o * D + r];
          const row = r * IN + 2 * d;
          for (let k = 0; k < g64.length; k++) v += h0.w[row + k] * g64[k];
          hid[r] = gelu(v);
        }
        linear(this.h1, hid, 0, logit, 0);
        out.set(logit, (s * N + o) * P);
      }
    }
    return out;
  }
}

type Box = NonNullable<ReturnType<typeof maskBox>>;

/** 8 pair features of (s, o), as train_head.pair_geom. */
export function pairGeom(s: Box, o: Box, inter: number, out: Float32Array) {
  const w = (b: Box) => Math.max(b.x1 - b.x0, 1e-3), h = (b: Box) => Math.max(b.y1 - b.y0, 1e-3);
  const ws = w(s), hs = h(s), wo = w(o), ho = h(o);
  const ref = (Math.sqrt(ws * hs) + Math.sqrt(wo * ho)) / 2;
  const ix = Math.max(0, Math.min(s.x1, o.x1) - Math.max(s.x0, o.x0));
  const iy = Math.max(0, Math.min(s.y1, o.y1) - Math.max(s.y0, o.y0));
  const bi = ix * iy;
  out[0] = ((o.x0 + o.x1) / 2 - (s.x0 + s.x1) / 2) / ref;
  out[1] = ((o.y0 + o.y1) / 2 - (s.y0 + s.y1) / 2) / ref;
  out[2] = Math.log(wo / ws);
  out[3] = Math.log(ho / hs);
  out[4] = bi / Math.max(ws * hs + wo * ho - bi, 1e-6);
  out[5] = inter / Math.max(s.count + o.count - inter, 1e-6);
  out[6] = inter / Math.max(s.count, 1e-6);
  out[7] = inter / Math.max(o.count, 1e-6);
}

/** Predicates that need motion: from the rules, since PSG is single images. */
const MOTION = ['approaching', 'moving away', 'chasing'] as const;

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Logits of the Tensor API head (wasm pipeline, ram/README.md section 6), by pipeline slot. */
export interface SlotLogits {
  logits: Float32Array;       // [M, M, P]: subject slot, object slot, predicate
  M: number;                  // slots (kMaxObjects)
  slot: Map<number, number>;  // object id -> slot
  ms: number;                 // time of the run + readback
}

/** Agreement of the Tensor API logits with the TS head on one frame (?relcheck=1). */
export interface RelCheck {
  n: number;        // objects
  maxDiff: number;  // max |logit difference| over pairs and predicates
  top1: number;     // pairs whose top predicate agrees
  pairs: number;
}

/**
 * The learned head as a RelationScorer: one forward per frame over all
 * objects (PSG predicates, sigmoid of the logits) plus the rule scorer's
 * motion predicates.
 */
export class LearnedScorer implements RelationScorer {
  readonly predicates: readonly string[];
  readonly symmetric = new Set(['beside', 'touching']);
  private readonly rules = new HeuristicScorer();
  private ext: SlotLogits | null = null;
  /** ms of the last forward (or Tensor API run). */
  lastMs = 0;
  /** Where the last frame's logits came from. */
  lastSource: 'ts' | 'tensorapi' = 'ts';
  /** Also run the TS head on Tensor API frames and record the agreement in `checks`. */
  check = false;
  readonly checks: RelCheck[] = [];

  constructor(readonly head: RelHead) {
    this.predicates = [...new Set([...head.preds, ...MOTION])];
  }

  /** Per-pair fallback (no scene context): the rules. */
  score(f: PairFeatures): Record<string, number> { return this.rules.score(f); }

  /** Uses these logits (instead of the TS forward) for the next scoreFrame only. */
  useLogits(l: SlotLogits | null) { this.ext = l; }

  scoreFrame(geoms: Geom[], pairs: Map<string, PairFeatures>): Map<string, Record<string, number>> {
    const out = new Map<string, Record<string, number>>();
    const ext = this.ext;
    this.ext = null;
    if (geoms.length < 2) return out;
    const N = geoms.length, P = this.head.preds.length;
    const slots = ext ? geoms.map((g) => ext.slot.get(g.id) ?? -1) : [];
    let logit: (s: number, o: number, p: number) => number;
    if (ext && ext.logits.length === ext.M * ext.M * P && slots.every((k) => k >= 0 && k < ext.M)) {
      this.lastMs = ext.ms;
      this.lastSource = 'tensorapi';
      logit = (s, o, p) => ext.logits[(slots[s] * ext.M + slots[o]) * P + p];
      if (this.check) this.compare(geoms, logit);
    } else {
      const t0 = performance.now();
      const logits = this.head.forward(geoms.map((g) => ({bits: g.bits, side: g.side, cls: g.cls})));
      this.lastMs = performance.now() - t0;
      this.lastSource = 'ts';
      logit = (s, o, p) => logits[(s * N + o) * P + p];
    }
    const off = this.head.meta.offsets?.[String(Math.min(6, N))] ?? 0;
    for (let s = 0; s < N; s++) {
      for (let o = 0; o < N; o++) {
        if (s === o) continue;
        const key = `${geoms[s].id},${geoms[o].id}`;
        const r: Record<string, number> = {};
        for (let p = 0; p < P; p++) {
          const v = sigmoid(logit(s, o, p) + off);
          if (v > 0.02) r[this.head.preds[p]] = v;
        }
        const f = pairs.get(key);
        if (f) {
          const m = this.rules.score(f);
          for (const k of MOTION) if (m[k] !== undefined) r[k] = Math.max(r[k] ?? 0, m[k]);
        }
        out.set(key, r);
      }
    }
    return out;
  }

  private compare(geoms: Geom[], logit: (s: number, o: number, p: number) => number) {
    const N = geoms.length, P = this.head.preds.length;
    const ref = this.head.forward(geoms.map((g) => ({bits: g.bits, side: g.side, cls: g.cls})));
    let maxDiff = 0, top1 = 0, pairs = 0;
    for (let s = 0; s < N; s++) {
      for (let o = 0; o < N; o++) {
        if (s === o) continue;
        let a = 0, b = 0;
        for (let p = 0; p < P; p++) {
          const x = logit(s, o, p), y = ref[(s * N + o) * P + p];
          maxDiff = Math.max(maxDiff, Math.abs(x - y));
          if (x > logit(s, o, a)) a = p;
          if (y > ref[(s * N + o) * P + b]) b = p;
        }
        pairs++;
        if (a === b) top1++;
      }
    }
    this.checks.push({n: N, maxDiff, top1, pairs});
    if (this.checks.length > 1000) this.checks.shift();
  }
}

/** Loads relhead.json + relhead.bin from `base` (e.g. "models/"). */
export async function loadRelHead(base: string): Promise<RelHead> {
  const [meta, bin] = await Promise.all([
    fetch(`${base}relhead.json`).then((r) => {
      if (!r.ok) throw new Error(`relhead.json: HTTP ${r.status}`);
      return r.json() as Promise<HeadMeta>;
    }),
    fetch(`${base}relhead.bin`).then((r) => {
      if (!r.ok) throw new Error(`relhead.bin: HTTP ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  return new RelHead(meta, new Float32Array(bin));
}
