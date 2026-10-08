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
"""Exports a trained geom_label relation head for the web demo (app/src/relhead.ts).

Writes
  app/public/models/relhead.json   config, PSG classes / predicates, tensor table
  app/public/models/relhead.bin    float32 weights, concatenated
  app/test/fixtures/relhead_fixture.json
                                   a few test images (masks, classes) with the
                                   PyTorch logits, so the TS port is checked
                                   against the reference.

  .venv/bin/python ram/export_web.py --ckpt artifacts/psg/heads/relhead_geom_label_web128.pt \
      --test artifacts/psg/feat_test.npz
"""
import argparse
import base64
import json
import os

import numpy as np
import torch

from train_head import RelHead, Split

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, '..', 'app')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', default=os.path.join(APP, 'public', 'models'))
    ap.add_argument('--fixture', default=os.path.join(APP, 'test', 'fixtures', 'relhead_fixture.json'))
    ap.add_argument('--fixture_images', type=int, default=4)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, weights_only=False)
    assert ck['variant'] == 'geom_label', 'the web port implements geom_label only'
    assert ck.get('geom_res') == 'full' and ck.get('box') == 'mask', \
        'train with --geom_res full --box mask (the geometry the browser has)'
    classes, preds, d = ck['classes'], ck['preds'], ck['d']
    model = RelHead(len(classes), len(preds), 'geom_label', d=d)
    model.load_state_dict(ck['state'])
    model.eval()

    # Weights: float32, concatenated in state_dict order.
    os.makedirs(a.out, exist_ok=True)
    table, chunks, off = {}, [], 0
    for name, t in model.state_dict().items():
        arr = t.detach().float().numpy().ravel()
        table[name] = {'offset': off, 'shape': list(t.shape)}
        chunks.append(arr)
        off += arr.size
    blob = np.concatenate(chunks).astype('<f4')
    blob.tofile(os.path.join(a.out, 'relhead.bin'))
    layer = model.enc.layers[0]
    meta = {
        'variant': 'geom_label', 'd': d, 'heads': layer.self_attn.num_heads, 'layers': len(model.enc.layers),
        'ff': layer.linear1.out_features, 'classes': classes, 'preds': preds, 'size': ck['size'],
        'tensors': table, 'params': int(blob.size), 'metrics': ck.get('metrics', {}),
    }
    with open(os.path.join(a.out, 'relhead.json'), 'w') as f:
        json.dump(meta, f)
    print(f'wrote relhead.json + relhead.bin: {blob.size / 1e6:.2f} M params, {blob.nbytes / 1e6:.1f} MB')

    # Fixture: up to 6 objects with real masks per image, PyTorch logits.
    test = Split(a.test, classes=classes, preds=preds, geom_res='full', box='mask')

    # Calibration: the multi-label loss puts the decision at logit 0, but PSG
    # annotates few pairs per image, so the head is conservative, and how
    # likely a pair is related depends on how many objects there are (PSG
    # images have ~12, the page tracks 2-6). So: run the head on random
    # 2..6-object subsets of the held-out images (the page's conditions) and
    # per object count pick the threshold on the best predicate per pair that
    # maximizes F1 for "this pair has an annotated relation". The page adds
    # -threshold[N] to the logits before the sigmoid.
    rng = np.random.default_rng(0)
    groups = {k: ([], []) for k in range(2, 7)}
    with torch.no_grad():
        for i in range(test.n):
            im = test.image(i)
            real = [k for k in range(len(im['cls'])) if im['gmask'][k].sum() >= 3]
            for _ in range(3):
                for n in range(2, min(6, len(real)) + 1):
                    keep = sorted(rng.choice(real, n, replace=False).tolist())
                    sub = {k: (v[keep] if k in ('ptr', 'pool', 'mask', 'gmask', 'box', 'cls') else v) for k, v in im.items()}
                    lg = model(sub).max(-1).values
                    idx = {k: j for j, k in enumerate(keep)}
                    pos = {(idx[s], idx[o]) for s, o, _ in im['rels'] if s in idx and o in idx}
                    for s in range(n):
                        for o in range(n):
                            if s != o:
                                groups[n][0].append(float(lg[s, o]))
                                groups[n][1].append((s, o) in pos)
    offsets, calib = {}, {}
    for n, (bl, hr) in groups.items():
        bl, hr = np.array(bl), np.array(hr)
        order = np.argsort(-bl)
        tp = np.cumsum(hr[order])
        prec = tp / np.arange(1, len(order) + 1)
        rec = tp / max(1, hr.sum())
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        k = int(np.argmax(f1))
        thr = float(bl[order[k]])
        offsets[n] = -thr
        calib[n] = {'threshold': round(thr, 3), 'f1': round(float(f1[k]), 3), 'precision': round(float(prec[k]), 3),
                    'recall': round(float(rec[k]), 3), 'pairs': int(len(bl)), 'base_rate': round(float(hr.mean()), 3)}
        print(f'calibration N={n}: threshold {thr:.2f}  F1 {f1[k]:.3f}  P {prec[k]:.3f}  R {rec[k]:.3f}  '
              f'base rate {hr.mean():.3f}  ({len(bl)} pairs)')
    meta['offsets'] = offsets  # by object count; counts > 6 use 6
    meta['calibration'] = calib
    with open(os.path.join(a.out, 'relhead.json'), 'w') as f:
        json.dump(meta, f)
    side = test.S // 4
    images = []
    for i in range(test.n):
        im = test.image(i)
        keep = [k for k in range(len(im['cls'])) if im['gmask'][k].sum() >= 3][:6]
        if len(keep) < 3:
            continue
        sub = {k: (v[keep] if k in ('ptr', 'pool', 'mask', 'gmask', 'box', 'cls') else v) for k, v in im.items()}
        with torch.no_grad():
            logits = model(sub)
        objs = []
        for j, k in enumerate(keep):
            bits = im['gmask'][k].numpy().astype(np.uint8).ravel()
            c = int(im['cls'][k])
            objs.append({'cls': classes[c - 1] if c else '', 'bits': base64.b64encode(bits.tobytes()).decode()})
        images.append({'side': side, 'objects': objs, 'logits': logits.numpy().round(5).ravel().tolist()})
        if len(images) >= a.fixture_images:
            break
    os.makedirs(os.path.dirname(a.fixture), exist_ok=True)
    with open(a.fixture, 'w') as f:
        json.dump({'preds': preds, 'images': images}, f)
    print(f'wrote {a.fixture}: {len(images)} images')


if __name__ == '__main__':
    main()
