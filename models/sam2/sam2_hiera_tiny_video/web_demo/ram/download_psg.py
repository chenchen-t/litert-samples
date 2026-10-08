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
"""Downloads PSG for the relation head, from the official release or the HF mirror.

--source official (recommended)
  The official PSG annotations (OpenPSG, https://github.com/Jingkang50/OpenPSG)
  are one file, psg.json, in the authors' SharePoint folder linked from the
  OpenPSG README ("We release the full dataset here", folder openpsg/data/psg/).
  SharePoint needs a browser, so download psg.json by hand to
  artifacts/psg/official/psg.json (or pass --psg_json). The images are COCO 2017;
  this script fetches only the ones it needs from the official COCO host
  (images.cocodataset.org) into artifacts/psg/official/coco/{train2017,val2017}/.
  Pass --coco_root to use an existing COCO copy instead (nothing is downloaded).
  The train / test split is the official one (psg.json "test_image_ids").

  .venv/bin/python ram/download_psg.py --source official --train_images 2000 --test_images 700  # ~ the pilot
  .venv/bin/python ram/download_psg.py --source official --train_images 0 --test_images 0       # everything

--source hf
  Unofficial parquet mirror JosephZ/psg_train_sg (46 shards, ~1k images each)
  and JosephZ/psg_test_sg (3 shards), images embedded. Used for the pilot.

  .venv/bin/python ram/download_psg.py --source hf --train_shards 2 --test_shards 1
"""
import argparse
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'artifacts', 'psg')
COCO_URL = 'http://images.cocodataset.org/'
PSG_README = 'https://github.com/Jingkang50/OpenPSG#updates'


def official_split(psg: dict, split: str) -> list:
    """Images of one official split, as OpenPSG's PanopticSceneGraphDataset:
    images with relations; test = psg['test_image_ids'], train = the rest."""
    test_ids = set(psg['test_image_ids'])
    data = [d for d in psg['data'] if d['relations']]
    return [d for d in data if (d['image_id'] in test_ids) == (split == 'test')]


def fetch(url: str, path: str) -> int:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, 'wb') as f:
        f.write(r.read())
    os.replace(tmp, path)
    return os.path.getsize(path)


def download_official(a):
    if not os.path.exists(a.psg_json):
        sys.exit(f'{a.psg_json} not found.\nDownload psg.json (folder data/psg/) from the official PSG release '
                 f'linked in the OpenPSG README ({PSG_README}) and save it there, or pass --psg_json.')
    with open(a.psg_json) as f:
        psg = json.load(f)
    print(f'psg.json: {len(psg["data"])} images, {len(psg["thing_classes"]) + len(psg["stuff_classes"])} classes, '
          f'{len(psg["predicate_classes"])} predicates, {len(psg["test_image_ids"])} test ids')
    if a.coco_root:
        missing = [d['file_name'] for s in ('train', 'test') for d in official_split(psg, s)
                   if not os.path.exists(os.path.join(a.coco_root, d['file_name']))]
        print(f'--coco_root {a.coco_root}: {len(missing)} PSG images missing' +
              (f' (e.g. {missing[0]})' if missing else ''))
        return
    root = os.path.join(a.out, 'official', 'coco')
    jobs = []
    for split, n in (('train', a.train_images), ('test', a.test_images)):
        items = official_split(psg, split)
        items = items[:n] if n else items
        print(f'{split}: {len(items)} images')
        jobs += [(COCO_URL + d['file_name'], os.path.join(root, d['file_name'])) for d in items]
    total = done = 0
    with ThreadPoolExecutor(8) as ex:
        for nbytes in ex.map(lambda j: fetch(*j), jobs):
            total += nbytes
            done += 1
            if done % 250 == 0 or done == len(jobs):
                print(f'  {done}/{len(jobs)} images ({total / 1e6:.1f} MB new)', flush=True)
    print(f'COCO images under {root}')


def download_hf(a):
    from huggingface_hub import hf_hub_download
    jobs = [('JosephZ/psg_train_sg', f'data/train-{i:05d}-of-00046.parquet', 'psg_train_sg')
            for i in range(min(a.train_shards, 46))]
    jobs += [('JosephZ/psg_test_sg', f'data/train-{i:05d}-of-00003.parquet', 'psg_test_sg')
             for i in range(min(a.test_shards, 3))]
    for repo, name, sub in jobs:
        path = hf_hub_download(repo, name, repo_type='dataset', local_dir=os.path.join(a.out, sub))
        print(f'{repo}/{name} -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['official', 'hf'], default='official')
    ap.add_argument('--out', default=OUT)
    # official
    ap.add_argument('--psg_json', default=os.path.join(OUT, 'official', 'psg.json'))
    ap.add_argument('--coco_root', default='', help='existing COCO 2017 root (train2017/, val2017/)')
    ap.add_argument('--train_images', type=int, default=2000, help='0 = all')
    ap.add_argument('--test_images', type=int, default=700, help='0 = all')
    # hf
    ap.add_argument('--train_shards', type=int, default=2)
    ap.add_argument('--test_shards', type=int, default=1)
    a = ap.parse_args()
    (download_official if a.source == 'official' else download_hf)(a)


if __name__ == '__main__':
    main()
