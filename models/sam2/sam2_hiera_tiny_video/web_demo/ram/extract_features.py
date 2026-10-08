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
"""Dumps SAM 2 per-object features on PSG for training a relation head.

Each PSG image is squashed to S x S (as the web pipeline does) and every
annotated object is prompted with its ground-truth box (labels 2 / 3, as the
demo prompts Gemma's boxes). From HF Sam2VideoModel at the demo's resolution
(tools/export_weights.model_at, the same weights / tables the .tflite is built
from) it keeps, per object, exactly what the web pipeline has on a prompt frame:

  ptr     [256]      object pointer (obj_ptr)            -> pipeline `ptr`
  mask    [S/4,S/4]  low-res mask logits (bit-packed)    -> pipeline `low_mask`
  pool    [256]      image embedding averaged over the mask (pix_raw, mask-pooled)
  score   []         object score logit
plus per image the image embedding pix_raw [S/16, S/16, 256] (fp16) for
union-region features, and the relations (subject, object, predicate).

Input: the official psg.json + COCO 2017 images (default, see download_psg.py)
or the HF parquet mirror (--source hf).

  .venv/bin/python ram/extract_features.py --split test --limit 700 --out artifacts/psg/feat_test.npz
  .venv/bin/python ram/extract_features.py --source hf --split test --shards 1 --out artifacts/psg/feat_test.npz
"""
import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
from export_weights import model_at  # noqa: E402
from transformers import Sam2VideoInferenceSession  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def cls_name(obj_id: str) -> str:
    """'tree-merged.3' -> 'tree-merged'."""
    return obj_id.rsplit('.', 1)[0]


def preprocess(img: Image.Image, S: int) -> torch.Tensor:
    x = np.asarray(img.convert('RGB').resize((S, S), Image.BILINEAR), np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].copy())


def find_pix_raw(cache: dict, g: int) -> torch.Tensor:
    """The S/16 image embedding [256, g, g] among the cached vision features."""
    cands = []
    for v in cache.values():
        for t in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(t, torch.Tensor):
                cands.append(t)
    for t in cands:
        s = tuple(t.shape)
        if s[-3:] == (256, g, g):
            return t.reshape(256, g, g)
        if len(s) == 3 and s[0] == g * g and s[-1] == 256:  # [HW, B, C]
            return t[:, 0].reshape(g, g, 256).permute(2, 0, 1)
    raise RuntimeError(f'no [256,{g},{g}] feature in cache: {[tuple(t.shape) for t in cands]}')


def images_hf(a):
    """(image_id, PIL image, objects [{id, cls, bbox xyxy px}], relations [{subject, object, predicate}])
    from the HF parquet mirror."""
    files = sorted(glob.glob(os.path.join(a.root, f'psg_{a.split}_sg', 'data', '*.parquet')))[:a.shards]
    if not files:
        sys.exit(f'no parquet shards under {a.root}/psg_{a.split}_sg/data')
    for f in files:
        for row in pq.read_table(f).to_pylist():
            objs = [dict(o, cls=cls_name(o['id'])) for o in json.loads(row['objects'])]
            yield row['image_id'], Image.open(io.BytesIO(row['image']['bytes'])), objs, json.loads(row['relationships'])


def images_official(a):
    """The same from the official psg.json + COCO 2017 images (download_psg.py --source official).
    Objects are psg.json 'annotations' (bbox XYXY_ABS, category_id into thing + stuff
    classes); relations are [subject, object, predicate] indices."""
    from download_psg import official_split
    with open(a.psg_json) as f:
        psg = json.load(f)
    names = psg['thing_classes'] + psg['stuff_classes']
    preds = psg['predicate_classes']
    coco = a.coco_root or os.path.join(a.root, 'official', 'coco')
    for d in official_split(psg, a.split):
        path = os.path.join(coco, d['file_name'])
        if not os.path.exists(path):
            continue  # not downloaded (download_psg.py --train_images / --test_images)
        objs = [{'id': k, 'cls': names[o['category_id']], 'bbox': o['bbox']} for k, o in enumerate(d['annotations'])]
        rels = [{'subject': s, 'object': o, 'predicate': preds[p]} for s, o, p in d['relations']]
        yield str(d['image_id']), Image.open(path), objs, rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'test'], required=True)
    ap.add_argument('--source', choices=['official', 'hf'], default='official')
    ap.add_argument('--root', default=os.path.join(HERE, '..', 'artifacts', 'psg'))
    ap.add_argument('--psg_json', default=os.path.join(HERE, '..', 'artifacts', 'psg', 'official', 'psg.json'),
                    help='official: psg.json')
    ap.add_argument('--coco_root', default='', help='official: COCO 2017 root (default <root>/official/coco)')
    ap.add_argument('--shards', type=int, default=1, help='hf: parquet shards to read')
    ap.add_argument('--limit', type=int, default=0, help='max images (0 = all)')
    ap.add_argument('--size', type=int, default=384)
    ap.add_argument('--max_objects', type=int, default=24)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    S, g, m4 = a.size, a.size // 16, a.size // 4
    source = images_official(a) if a.source == 'official' else images_hf(a)
    model = model_at(S)
    torch.set_grad_enabled(False)

    img_ids, obj_off, pix = [], [0], []
    o_cls, o_ptr, o_pool, o_mask, o_box, o_score = [], [], [], [], [], []
    rels = []  # (image, subject_local, object_local, predicate name)
    n_img, t0 = 0, time.time()
    for image_id, img, objs, rs in source:
        if a.limit and n_img >= a.limit:
            break
        img = img.convert('RGB')
        W, H = img.size
        # Objects in relations first, then the rest, capped.
        in_rel = {r['subject'] for r in rs} | {r['object'] for r in rs}
        objs = sorted(objs, key=lambda o: o['id'] not in in_rel)[:a.max_objects]
        local = {o['id']: k for k, o in enumerate(objs)}
        if len(objs) < 2:
            continue

        sess = Sam2VideoInferenceSession(video_height=S, video_width=S, dtype=torch.float32)
        for k, o in enumerate(objs):
            x0, y0, x1, y1 = o['bbox']
            box = [[min(max(x0 / W * S, 0), S - 1), min(max(y0 / H * S, 0), S - 1)],
                   [min(max(x1 / W * S, 0), S - 1), min(max(y1 / H * S, 0), S - 1)]]
            oi = sess.obj_id_to_idx(k + 1)
            sess.add_point_inputs(oi, 0, {'point_coords': torch.tensor([[box]], dtype=torch.float32),
                                          'point_labels': torch.tensor([[[2, 3]]], dtype=torch.int32)})
        sess.obj_with_new_inputs = list(range(1, len(objs) + 1))
        model(inference_session=sess, frame=preprocess(img, S), run_mem_encoder=False)
        raw = find_pix_raw(sess.cache.get_vision_features(0), g)  # [256, g, g]
        for k, o in enumerate(objs):
            out = sess.output_dict_per_obj[sess.obj_id_to_idx(k + 1)]['cond_frame_outputs'][0]
            mask = out['pred_masks'].reshape(m4, m4)
            w = torch.sigmoid(mask)[None, None]
            w = torch.nn.functional.avg_pool2d(w, m4 // g)[0, 0]  # [g, g]
            pool = (raw * w).sum((1, 2)) / w.sum().clamp_min(1e-3)
            x0, y0, x1, y1 = o['bbox']
            o_cls.append(o['cls'])
            o_ptr.append(out['object_pointer'].reshape(256).numpy().astype(np.float16))
            o_pool.append(pool.numpy().astype(np.float16))
            o_mask.append(np.packbits((mask > 0).numpy().reshape(-1)))
            o_box.append(np.array([x0 / W, y0 / H, x1 / W, y1 / H], np.float32))
            o_score.append(float(out['object_score_logits'].reshape(-1)[0]))
        for r in rs:
            if r['subject'] in local and r['object'] in local:
                rels.append((n_img, local[r['subject']], local[r['object']], r['predicate']))
        pix.append(raw.permute(1, 2, 0).numpy().astype(np.float16))
        img_ids.append(image_id)
        obj_off.append(len(o_cls))
        n_img += 1
        if n_img % 50 == 0:
            dt = time.time() - t0
            print(f'{n_img} images, {len(o_cls)} objects, {len(rels)} relations, {dt / n_img:.2f} s/img', flush=True)

    np.savez(a.out,
             image_id=np.array(img_ids), obj_offset=np.array(obj_off, np.int64), pix_raw=np.stack(pix),
             cls=np.array(o_cls), ptr=np.stack(o_ptr), pool=np.stack(o_pool), mask=np.stack(o_mask),
             box=np.stack(o_box), score=np.array(o_score, np.float32),
             rel=np.array([(i, s, o) for i, s, o, _ in rels], np.int64).reshape(-1, 3),
             rel_pred=np.array([p for *_, p in rels]), size=np.array(S))
    print(f'wrote {a.out}: {n_img} images, {len(o_cls)} objects, {len(rels)} relations '
          f'in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
