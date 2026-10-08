# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts the exported relation head to safetensors for the Tensor API graph.

Reads what export_web.py wrote (app/public/models/relhead.{json,bin} and
app/test/fixtures/relhead_fixture.json) and writes

  app/public/models/relhead.safetensors   weights under the PyTorch names (F32)
                                          + metadata (d, heads, layers, classes, preds)
  cc/testdata/relhead.safetensors         the same, for the C++ test
  cc/testdata/relhead_fixture.safetensors per fixture image k, padded to 6 slots:
      img{k}_masks  [6, S/4, S/4] U8   mask bits (1 inside; empty slots all 0)
      img{k}_cls    [6]           F32  PSG class index (0 = unknown)
      img{k}_valid  [6]           F32  1 = object present
      img{k}_logits [6, 6, P]     F32  PyTorch logits (rows / cols >= n are 0)

  .venv/bin/python ram/export_tensorapi.py
"""
import argparse
import base64
import json
import os

import numpy as np
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
K = 6  # kMaxObjects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=os.path.join(ROOT, 'app', 'public', 'models'))
    ap.add_argument('--fixture', default=os.path.join(ROOT, 'app', 'test', 'fixtures', 'relhead_fixture.json'))
    ap.add_argument('--testdata', default=os.path.join(ROOT, 'cc', 'testdata'))
    ckpt = os.path.join(ROOT, 'artifacts', 'psg', 'heads', 'relhead_geom_label_web128.pt')
    ap.add_argument('--ckpt', default=ckpt if os.path.exists(ckpt) else '',
                    help='PyTorch checkpoint for the fewer-object reference cases (needs torch)')
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.models, 'relhead.json')))
    blob = np.fromfile(os.path.join(a.models, 'relhead.bin'), '<f4')
    tensors = {}
    for name, e in meta['tensors'].items():
        n = int(np.prod(e['shape']))
        tensors[name] = blob[e['offset']:e['offset'] + n].reshape(e['shape']).copy()
    md = {k: json.dumps(meta[k]) for k in ('d', 'heads', 'layers', 'ff', 'classes', 'preds', 'size')}
    md['offsets'] = json.dumps(meta.get('offsets', {}))
    for path in (os.path.join(a.models, 'relhead.safetensors'), os.path.join(a.testdata, 'relhead.safetensors')):
        save_file(tensors, path, metadata=md)
        print(f'wrote {path}: {len(tensors)} tensors, {blob.size / 1e6:.2f} M params')

    fx = json.load(open(a.fixture))
    cidx = {c: i + 1 for i, c in enumerate(meta['classes'])}
    P = len(fx['preds'])
    cases = []  # (side, [(bits, cls)], logits [n, n, P])
    for im in fx['images']:
        objs = [(np.frombuffer(base64.b64decode(o['bits']), np.uint8).reshape(im['side'], im['side']), o['cls'])
                for o in im['objects']]
        n = len(objs)
        cases.append((im['side'], objs, np.array(im['logits'], np.float32).reshape(n, n, P)))
    if a.ckpt:
        # Fewer objects than slots (the page's usual case): the first 2..5 objects of each
        # image, reference logits from the PyTorch head.
        import torch
        from train_head import RelHead
        ck = torch.load(a.ckpt, weights_only=False)
        model = RelHead(len(ck['classes']), len(ck['preds']), 'geom_label', d=ck['d'])
        model.load_state_dict(ck['state'])
        model.eval()
        for side, objs, _ in list(cases):
            for n in range(2, min(6, len(objs))):
                sub = objs[:n]
                gm = torch.tensor(np.stack([b for b, _ in sub]), dtype=torch.float32)
                boxes = []
                for b, _ in sub:
                    r, c = np.flatnonzero(b.any(1)), np.flatnonzero(b.any(0))
                    boxes.append([c[0] / side, r[0] / side, (c[-1] + 1) / side, (r[-1] + 1) / side])
                im = {'gmask': gm, 'box': torch.tensor(boxes, dtype=torch.float32),
                      'cls': torch.tensor([cidx.get(c, 0) for _, c in sub])}
                with torch.no_grad():
                    cases.append((side, sub, model(im).numpy()))
    out = {}
    for k, (side, objs, ref) in enumerate(cases):
        n = len(objs)
        masks = np.zeros((K, side, side), np.uint8)
        cls = np.zeros(K, np.float32)
        valid = np.zeros(K, np.float32)
        for j, (bits, c) in enumerate(objs):
            masks[j] = bits > 0
            cls[j] = cidx.get(c, 0)
            valid[j] = 1
        logits = np.zeros((K, K, P), np.float32)
        logits[:n, :n] = ref
        out.update({f'img{k}_masks': masks, f'img{k}_cls': cls, f'img{k}_valid': valid, f'img{k}_logits': logits})
    path = os.path.join(a.testdata, 'relhead_fixture.safetensors')
    save_file(out, path, metadata={'images': str(len(cases)), 'side': str(cases[0][0])})
    print(f'wrote {path}: {len(cases)} cases (object counts {[len(c[1]) for c in cases]})')


if __name__ == '__main__':
    main()
