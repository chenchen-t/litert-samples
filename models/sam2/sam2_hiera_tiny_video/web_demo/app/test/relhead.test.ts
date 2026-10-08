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

// The TS relation head against the PyTorch reference (ram/export_web.py fixture).

import {existsSync, readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import {type HeadMeta, RelHead} from '../src/relhead';

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));
const META = here('../public/models/relhead.json');
const BIN = here('../public/models/relhead.bin');
const FIXTURE = here('./fixtures/relhead_fixture.json');
const have = existsSync(META) && existsSync(BIN) && existsSync(FIXTURE);

interface Fixture {
  preds: string[];
  images: Array<{side: number; objects: Array<{cls: string; bits: string}>; logits: number[]}>;
}

describe.skipIf(!have)('RelHead (TS) vs PyTorch', () => {
  const load = () => {
    const meta = JSON.parse(readFileSync(META, 'utf8')) as HeadMeta;
    const buf = readFileSync(BIN);
    const data = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    return new RelHead(meta, data);
  };

  it('matches the reference logits', () => {
    const head = load();
    const fx = JSON.parse(readFileSync(FIXTURE, 'utf8')) as Fixture;
    expect(head.preds).toEqual(fx.preds);
    expect(fx.images.length).toBeGreaterThan(0);
    for (const im of fx.images) {
      const objs = im.objects.map((o) => ({cls: o.cls, side: im.side, bits: new Uint8Array(Buffer.from(o.bits, 'base64'))}));
      const got = head.forward(objs);
      const N = objs.length, P = head.preds.length;
      let maxErr = 0;
      for (let s = 0; s < N; s++) {
        for (let o = 0; o < N; o++) {
          if (s === o) continue;
          for (let p = 0; p < P; p++) {
            const i = (s * N + o) * P + p;
            maxErr = Math.max(maxErr, Math.abs(got[i] - im.logits[i]));
          }
        }
      }
      expect(maxErr).toBeLessThan(2e-3);
    }
  });

  it('unknown classes still score, and is fast for 6 objects', () => {
    const head = load();
    const fx = JSON.parse(readFileSync(FIXTURE, 'utf8')) as Fixture;
    const im = fx.images[0];
    const objs = im.objects.map((o) => ({cls: '', side: im.side, bits: new Uint8Array(Buffer.from(o.bits, 'base64'))}));
    const out = head.forward(objs);
    expect(out.every((v, i) => Number.isFinite(v) || Math.floor(i / head.preds.length) % (objs.length + 1) === 0))
        .toBe(true);
    for (let k = 0; k < 20; k++) head.forward(objs);
    const t0 = performance.now();
    for (let k = 0; k < 50; k++) head.forward(objs);
    const ms = (performance.now() - t0) / 50;
    expect(ms).toBeLessThan(10);
    console.info(`relation head: ${objs.length} objects in ${ms.toFixed(2)} ms`);
  });
});
