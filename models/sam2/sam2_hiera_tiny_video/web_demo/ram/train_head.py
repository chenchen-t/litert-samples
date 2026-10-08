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
"""Trains / evaluates a small relation head on SAM 2 features (extract_features.py).

Model (RAM-style, sized for the web pipeline: <= 6 objects, ~1 ms):
  object token = MLP([ptr 256, mask-pooled pix_raw 256, box geometry 9, label emb 64])
  2-layer transformer over the image's objects (d=256)
  pair head    = MLP([tok_s, tok_o, pair geometry 8, union-pooled pix_raw 256]) -> P predicates
Loss: multi-label categorical cross-entropy (as RAM), every ordered pair; pairs
without a relation only push their scores down.

Variants (--variant):
  full        everything above
  nolabel     no label embedding (works with any free-form Gemma label)
  geom_label  only box/pair geometry + labels (learned "option A", no SAM 2 features)
and the frequency baseline P(predicate | subject class, object class).

Metrics on the test features (ground-truth objects, PredCls-style):
  acc@1 / acc@5  predicate accuracy for annotated pairs (what the overlay shows)
  R@K / mR@K     triplet recall in the top K over all pairs (one predicate per pair),
                 mR = mean over predicates (rare predicates count equally)

  .venv/bin/python ram/train_head.py --train artifacts/psg/feat_train.npz --test artifacts/psg/feat_test.npz
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- data

class Split:
    """One feature file. geom_res='full' computes geometry on the S/4 masks the
    browser reads (else on the S/16 grid); box='mask' uses the mask's bounding
    box (what the browser has while tracking) instead of the annotated box."""

    def __init__(self, path, classes=None, preds=None, geom_res='grid', box='gt'):
        d = np.load(path)
        self.S = int(d['size'])
        self.off = d['obj_offset']
        self.n = len(self.off) - 1
        self.ptr = torch.from_numpy(d['ptr'].astype(np.float32))
        self.pool = torch.from_numpy(d['pool'].astype(np.float32))
        self.pix = d['pix_raw']  # [I, g, g, 256] fp16
        m4 = self.S // 4
        g = self.S // 16
        bits = np.unpackbits(d['mask'], axis=1)[:, :m4 * m4].reshape(-1, m4, m4)  # uint8
        masks = torch.from_numpy(bits.astype(np.float32))
        self.mask24 = F.avg_pool2d(masks[:, None], m4 // g)[:, 0]  # [O, g, g] fractions
        self.mbits = torch.from_numpy(bits) if geom_res == 'full' else None
        self.box = torch.from_numpy(d['box'])
        if box == 'mask':
            ys, xs = bits.any(2), bits.any(1)  # [O, m4] rows / cols with mask
            mb = self.box.clone()
            for k in range(len(bits)):
                r, c = np.flatnonzero(ys[k]), np.flatnonzero(xs[k])
                if len(r) and len(c):  # empty masks keep the annotated box
                    mb[k] = torch.tensor([c[0] / m4, r[0] / m4, (c[-1] + 1) / m4, (r[-1] + 1) / m4])
            self.box = mb
        cls = list(d['cls'])
        self.classes = classes or sorted(set(cls))
        cidx = {c: i + 1 for i, c in enumerate(self.classes)}  # 0 = unknown
        self.cls = torch.tensor([cidx.get(c, 0) for c in cls])
        rp = list(d['rel_pred'])
        self.preds = preds or sorted(set(rp))
        pidx = {p: i for i, p in enumerate(self.preds)}
        self.rels = [[] for _ in range(self.n)]
        for (i, s, o), p in zip(d['rel'], rp):
            if p in pidx:
                self.rels[i].append((int(s), int(o), pidx[p]))

    def image(self, i):
        a, b = self.off[i], self.off[i + 1]
        gmask = self.mbits[a:b].float() if self.mbits is not None else self.mask24[a:b]
        return dict(ptr=self.ptr[a:b], pool=self.pool[a:b], mask=self.mask24[a:b], gmask=gmask, box=self.box[a:b],
                    cls=self.cls[a:b], pix=torch.from_numpy(self.pix[i].astype(np.float32)), rels=self.rels[i])


def obj_geom(box, mask):
    """[N, 9]: box x0 y0 x1 y1, centre, size, mask area."""
    x0, y0, x1, y1 = box.unbind(-1)
    area = mask.mean((1, 2))
    return torch.stack([x0, y0, x1, y1, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, area], -1)


def pair_geom(box, mask):
    """[N, N, 8] for (s, o): centre offset / size, log size ratios, box IoU, mask IoU, containment both ways."""
    x0, y0, x1, y1 = box.unbind(-1)
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0).clamp_min(1e-3), (y1 - y0).clamp_min(1e-3)
    sz = (w * h).sqrt()
    ref = (sz[:, None] + sz[None]) / 2
    dx = (cx[None] - cx[:, None]) / ref
    dy = (cy[None] - cy[:, None]) / ref
    lw = (w[None] / w[:, None]).log()
    lh = (h[None] / h[:, None]).log()
    ix = (torch.min(x1[:, None], x1[None]) - torch.max(x0[:, None], x0[None])).clamp_min(0)
    iy = (torch.min(y1[:, None], y1[None]) - torch.max(y0[:, None], y0[None])).clamp_min(0)
    inter = ix * iy
    biou = inter / (w * h)[:, None].add((w * h)[None]).sub(inter).clamp_min(1e-6)
    m = mask.flatten(1)
    mi = m @ m.T
    ma = m.sum(1)
    miou = mi / (ma[:, None] + ma[None] - mi).clamp_min(1e-6)
    cs = mi / ma[:, None].clamp_min(1e-6)  # fraction of s inside o
    co = mi / ma[None].clamp_min(1e-6)     # fraction of o inside s
    return torch.stack([dx, dy, lw, lh, biou, miou, cs, co], -1)


def union_pool(pix, mask):
    """[N, N, 256]: pix_raw averaged over max(mask_s, mask_o)."""
    u = torch.max(mask[:, None], mask[None])  # [N, N, g, g]
    num = torch.einsum('stxy,xyc->stc', u, pix)
    return num / u.sum((2, 3))[..., None].clamp_min(1e-3)


# ---------------------------------------------------------------- model

class RelHead(nn.Module):
    def __init__(self, n_cls, n_pred, variant='full', d=256):
        super().__init__()
        self.variant = variant
        self.visual = variant in ('full', 'nolabel')
        self.labels = variant in ('full', 'geom_label')
        din = 9 + (512 if self.visual else 0) + (64 if self.labels else 0)
        if self.labels:
            self.emb = nn.Embedding(n_cls + 1, 64)
        self.inp = nn.Sequential(nn.Linear(din, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(0.1))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.pg = nn.Sequential(nn.Linear(8, 64), nn.GELU())
        if self.visual:
            self.up = nn.Sequential(nn.Linear(256, 128), nn.GELU())
        dp = 2 * d + 64 + (128 if self.visual else 0)
        self.head = nn.Sequential(nn.Linear(dp, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, n_pred))

    def forward(self, im, label_dropout=0.0):
        mask, box = im['gmask'], im['box']
        parts = [obj_geom(box, mask)]
        if self.visual:
            parts += [im['ptr'] / 10.0, im['pool']]
        if self.labels:
            cls = im['cls']
            if self.training and label_dropout:
                cls = torch.where(torch.rand(cls.shape) < label_dropout, torch.zeros_like(cls), cls)
            parts.append(self.emb(cls))
        tok = self.enc(self.inp(torch.cat(parts, -1))[None])[0]  # [N, d]
        N = tok.shape[0]
        pair = [tok[:, None].expand(N, N, -1), tok[None].expand(N, N, -1), self.pg(pair_geom(box, mask))]
        if self.visual:
            pair.append(self.up(union_pool(im['pix'], im['mask'])))
        return self.head(torch.cat(pair, -1))  # [N, N, P]


def ml_ce(logits, target):
    """Multi-label categorical cross-entropy (Su 2020, as RAM): rows = pairs."""
    pos = target > 0
    neg_l = logits.masked_fill(pos, -1e4)
    pos_l = (-logits).masked_fill(~pos, -1e4)
    zero = torch.zeros_like(logits[..., :1])
    return (torch.logsumexp(torch.cat([neg_l, zero], -1), -1) + torch.logsumexp(torch.cat([pos_l, zero], -1), -1)).mean()


# ---------------------------------------------------------------- eval

def evaluate(score_fn, split, n_pred, ks=(20, 50, 100)):
    hits = {k: 0 for k in ks}
    hits_p = {k: Counter() for k in ks}
    tot_p = Counter()
    total = 0
    a1 = a5 = npair = 0
    for i in range(split.n):
        im = split.image(i)
        if not im['rels']:
            continue
        sc = score_fn(im)  # [N, N, P]
        N = sc.shape[0]
        sc = sc.clone()
        sc[torch.arange(N), torch.arange(N)] = -1e9
        gt = set(im['rels'])
        # Predicate accuracy for annotated pairs (multi-label: any GT predicate counts).
        bypair = {}
        for s, o, p in gt:
            bypair.setdefault((s, o), set()).add(p)
        for (s, o), ps in bypair.items():
            top = sc[s, o].topk(5).indices.tolist()
            a1 += top[0] in ps
            a5 += bool(ps & set(top))
            npair += 1
        # Graph-constrained triplets: best predicate per pair.
        best, bp = sc.max(-1)
        order = best.flatten().argsort(descending=True)
        for k in ks:
            pred = {(int(j // N), int(j % N), int(bp.flatten()[j])) for j in order[:k]}
            h = gt & pred
            hits[k] += len(h)
            for (_, _, p) in h:
                hits_p[k][p] += 1
        for (_, _, p) in gt:
            tot_p[p] += 1
        total += len(gt)
    res = {'acc@1': a1 / npair, 'acc@5': a5 / npair}
    for k in ks:
        res[f'R@{k}'] = hits[k] / total
        res[f'mR@{k}'] = float(np.mean([hits_p[k][p] / tot_p[p] for p in tot_p]))
    return res


def frequency_baseline(train, n_pred):
    cnt = {}
    prior = torch.zeros(n_pred)
    for i in range(train.n):
        cls = train.cls[train.off[i]:train.off[i + 1]]
        for s, o, p in train.rels[i]:
            key = (int(cls[s]), int(cls[o]))
            cnt.setdefault(key, torch.zeros(n_pred))[p] += 1
            prior[p] += 1
    prior = prior / prior.sum()

    def score(im):
        cls = im['cls'].tolist()
        N = len(cls)
        out = torch.empty(N, N, n_pred)
        for s in range(N):
            for o in range(N):
                c = cnt.get((cls[s], cls[o]))
                # Backoff to the prior; scale by how often the class pair has relations at all.
                out[s, o] = (c + prior).log() if c is not None else (prior * 1e-3).log()
        return out
    return score


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--variants', default='full,nolabel,geom_label')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--label_dropout', type=float, default=0.2)
    ap.add_argument('--d', type=int, default=256, help='model width')
    ap.add_argument('--geom_res', choices=['grid', 'full'], default='grid',
                    help='geometry on the S/16 grid or the S/4 masks (the browser uses S/4)')
    ap.add_argument('--box', choices=['gt', 'mask'], default='gt',
                    help='annotated boxes or the masks\' bounding boxes (the browser has the latter)')
    ap.add_argument('--tag', default='', help='suffix for the saved heads / results')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'psg', 'heads'))
    a = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)

    train = Split(a.train, geom_res=a.geom_res, box=a.box)
    test = Split(a.test, classes=train.classes, preds=train.preds, geom_res=a.geom_res, box=a.box)
    P = len(train.preds)
    n_rel = sum(len(r) for r in train.rels)
    print(f'train: {train.n} images, {len(train.cls)} objects, {n_rel} relations; '
          f'test: {test.n} images, {sum(len(r) for r in test.rels)} relations; '
          f'{len(train.classes)} classes, {P} predicates')

    results = {'frequency': evaluate(frequency_baseline(train, P), test, P)}
    print('frequency   ', json.dumps({k: round(v, 3) for k, v in results['frequency'].items()}))

    for variant in a.variants.split(','):
        model = RelHead(len(train.classes), P, variant, d=a.d)
        nparam = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * train.n)
        t0 = time.time()
        for ep in range(a.epochs):
            model.train()
            tot = 0.0
            for i in torch.randperm(train.n).tolist():
                im = train.image(i)
                N = len(im['cls'])
                target = torch.zeros(N, N, P)
                for s, o, p in im['rels']:
                    target[s, o, p] = 1
                logits = model(im, a.label_dropout)
                off = ~torch.eye(N, dtype=torch.bool)
                loss = ml_ce(logits[off], target[off])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            if ep % 5 == 4 or ep == a.epochs - 1:
                model.eval()
                with torch.no_grad():
                    r = evaluate(lambda im: model(im), test, P)
                print(f'{variant:11s} ep {ep + 1:2d} loss {tot / train.n:.3f} ' +
                      ' '.join(f'{k} {v:.3f}' for k, v in r.items()) + f'  ({time.time() - t0:.0f} s)', flush=True)
        model.eval()
        with torch.no_grad():
            results[variant] = evaluate(lambda im: model(im), test, P)
            # Latency at the demo's object budget (6 objects), CPU, one thread.
            torch.set_num_threads(1)
            im = test.image(0)
            im6 = {k: (v[:6] if k in ('ptr', 'pool', 'mask', 'gmask', 'box', 'cls') else v) for k, v in im.items()}
            for _ in range(5):
                model(im6)
            t1 = time.perf_counter()
            for _ in range(50):
                model(im6)
            ms = (time.perf_counter() - t1) / 50 * 1000
            torch.set_num_threads(os.cpu_count() or 4)
        results[variant]['params_M'] = nparam / 1e6
        results[variant]['cpu_ms_6obj'] = ms
        torch.save({'state': model.state_dict(), 'variant': variant, 'classes': train.classes, 'preds': train.preds,
                    'd': a.d, 'geom_res': a.geom_res, 'box': a.box, 'size': train.S, 'metrics': results[variant]},
                   os.path.join(a.out, f'relhead_{variant}{a.tag}.pt'))
        print(f'{variant:11s} final', json.dumps({k: round(v, 3) for k, v in results[variant].items()}))

    with open(os.path.join(a.out, f'results{a.tag}.json'), 'w') as f:
        json.dump({'train_images': train.n, 'test_images': test.n, 'predicates': train.preds, 'results': results}, f, indent=1)
    print('\n| model | acc@1 | acc@5 | R@20 | R@50 | mR@50 | params | CPU ms (6 obj) |')
    print('|---|---|---|---|---|---|---|---|')
    for k, r in results.items():
        print(f"| {k} | {r['acc@1']:.3f} | {r['acc@5']:.3f} | {r['R@20']:.3f} | {r['R@50']:.3f} | {r['mR@50']:.3f} | "
              f"{r.get('params_M', 0):.2f} M | {r.get('cpu_ms_6obj', 0):.2f} |")


if __name__ == '__main__':
    main()
