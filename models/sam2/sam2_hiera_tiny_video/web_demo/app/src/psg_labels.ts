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

// Gemma's free-text labels -> the 133 PSG classes the relation head knows.
//
// 1. A table: exact / cleaned names, synonyms ("player" -> person), plurals,
//    adjectives dropped ("red car" -> car), then the head noun.
// 2. Labels the table misses go to Gemma once, as a text-only question with
//    the class list (gemmaPrompt / parseGemmaReply).
// 3. Answers are checked: anything that is not one of the classes is unknown
//    (class 0, which the head was trained to handle via label dropout).

/** PSG (COCO panoptic) class names, as in the training data. */
export const PSG_CLASSES = [
  'airplane', 'apple', 'backpack', 'banana', 'banner', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench',
  'bicycle', 'bird', 'blanket', 'boat', 'book', 'bottle', 'bowl', 'bridge', 'broccoli', 'building-other-merged', 'bus',
  'cabinet-merged', 'cake', 'car', 'cardboard', 'carrot', 'cat', 'ceiling-merged', 'cell phone', 'chair', 'clock',
  'couch', 'counter', 'cow', 'cup', 'curtain', 'dining table', 'dirt-merged', 'dog', 'donut', 'door-stuff', 'elephant',
  'fence-merged', 'fire hydrant', 'floor-other-merged', 'floor-wood', 'flower', 'food-other-merged', 'fork', 'frisbee',
  'fruit', 'giraffe', 'grass-merged', 'gravel', 'hair drier', 'handbag', 'horse', 'hot dog', 'house', 'keyboard',
  'kite', 'knife', 'laptop', 'light', 'microwave', 'mirror-stuff', 'motorcycle', 'mountain-merged', 'mouse', 'net',
  'orange', 'oven', 'paper-merged', 'parking meter', 'pavement-merged', 'person', 'pillow', 'pizza', 'platform',
  'playingfield', 'potted plant', 'railroad', 'refrigerator', 'remote', 'river', 'road', 'rock-merged', 'roof',
  'rug-merged', 'sand', 'sandwich', 'scissors', 'sea', 'sheep', 'shelf', 'sink', 'skateboard', 'skis',
  'sky-other-merged', 'snow', 'snowboard', 'spoon', 'sports ball', 'stairs', 'stop sign', 'suitcase', 'surfboard',
  'table-merged', 'teddy bear', 'tennis racket', 'tent', 'tie', 'toaster', 'toilet', 'toothbrush', 'towel',
  'traffic light', 'train', 'tree-merged', 'truck', 'tv', 'umbrella', 'vase', 'wall-brick', 'wall-other-merged',
  'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'wine glass', 'zebra',
] as const;

/** Readable name of a PSG class: "grass-merged" -> "grass", "wall-brick" -> "brick wall". */
export function cleanName(cls: string): string {
  const base = cls.replace(/-(merged|other|stuff)/g, '');
  const m = /^(wall|floor|window)-(\w+)$/.exec(base);
  if (m) return m[2] === 'blind' ? 'window blind' : `${m[2]} ${m[1]}`;
  return base.replace(/-/g, ' ');
}

const BY_CLEAN = new Map<string, string>();
for (const c of PSG_CLASSES) {
  BY_CLEAN.set(c, c);
  BY_CLEAN.set(cleanName(c), c);
}

/** Free-text words -> PSG class (singular, lower case). */
const SYNONYMS: Record<string, string> = {};
const syn = (cls: string, words: string) => {
  for (const w of words.split(',')) SYNONYMS[w.trim()] = cls;
};
syn('person', 'people, man, woman, boy, girl, child, kid, baby, toddler, adult, player, footballer, soccer player, ' +
    'athlete, goalkeeper, goalie, keeper, referee, umpire, coach, human, guy, lady, gentleman, dancer, runner, ' +
    'rider, cyclist, skier, skater, skateboarder, surfer, snowboarder, pedestrian, spectator, fan, worker, chef, ' +
    'student, teacher, someone, individual, figure, batter, pitcher, catcher, tennis player, swimmer, driver, ' +
    'passenger, police officer, officer, soldier, farmer, doctor, nurse, mother, father, son, daughter, teenager, ' +
    'player in red, player in white, player in blue, defender, striker, attacker, midfielder, crowd, audience');
syn('sports ball', 'ball, football, soccer ball, soccerball, basketball, volleyball, tennis ball, baseball, ' +
    'golf ball, rugby ball, beach ball');
syn('playingfield', 'field, football field, soccer field, pitch, football pitch, soccer pitch, court, ' +
    'tennis court, basketball court, baseball field, stadium, turf, playing field, sports field');
syn('grass-merged', 'lawn, meadow, grassland, grass field');
syn('dog', 'puppy, pup, hound');
syn('cat', 'kitten, kitty');
syn('horse', 'pony, foal');
syn('cow', 'cattle, calf, bull, ox');
syn('sheep', 'lamb');
syn('bird', 'pigeon, seagull, gull, duck, goose, parrot, crow, sparrow');
syn('bicycle', 'bike, cycle');
syn('motorcycle', 'motorbike, scooter, moped');
syn('car', 'automobile, vehicle, taxi, cab, suv, sedan, van, jeep');
syn('truck', 'lorry, pickup, pickup truck');
syn('airplane', 'plane, aeroplane, jet, aircraft');
syn('boat', 'ship, kayak, canoe, sailboat, yacht');
syn('cell phone', 'phone, mobile phone, smartphone, cellphone, mobile');
syn('tv', 'television, monitor, screen');
syn('laptop', 'computer, notebook computer');
syn('couch', 'sofa');
syn('dining table', 'table, desk, kitchen table');
syn('potted plant', 'plant, houseplant');
syn('tree-merged', 'trees, bush, shrub, palm tree');
syn('sky-other-merged', 'clouds, cloud');
syn('road', 'street, highway, lane');
syn('pavement-merged', 'sidewalk, footpath, walkway');
syn('sea', 'ocean, waves, wave');
syn('water-other', 'water, lake, pool, pond');
syn('river', 'stream, creek');
syn('mountain-merged', 'hill, hills');
syn('rock-merged', 'stone, boulder');
syn('building-other-merged', 'building, skyscraper, tower');
syn('house', 'home, cottage, cabin');
syn('wall-other-merged', 'wall');
syn('floor-other-merged', 'floor, ground');
syn('dirt-merged', 'mud, soil, earth');
syn('net', 'goal net, goal, tennis net');
syn('handbag', 'bag, purse');
syn('backpack', 'rucksack, knapsack');
syn('cup', 'mug');
syn('wine glass', 'glass');
syn('book', 'books, magazine');
syn('light', 'lamp, streetlight, street light');
syn('railroad', 'railway, train track, tracks');
syn('tennis racket', 'racket, racquet');
syn('baseball bat', 'bat');
syn('baseball glove', 'glove, mitt');
syn('teddy bear', 'stuffed animal, teddy');
syn('rug-merged', 'rug, carpet, mat');
syn('remote', 'remote control, controller');

const ADJECTIVES = new Set(('a an the this that some two three four many several small little big large tall short ' +
    'young old left right front back far near red blue green yellow white black gray grey brown orange pink purple ' +
    'dark light striped wooden metal plastic green-shirted other another main first second third standing sitting ' +
    'running walking moving distant nearby').split(' '));

/** Singular of one word (rough English rules plus a few irregulars). */
function singular(w: string): string {
  const irr: Record<string, string> = {people: 'person', men: 'man', women: 'woman', children: 'child',
    feet: 'foot', geese: 'goose', mice: 'mouse', teeth: 'tooth', persons: 'person', skis: 'skis', stairs: 'stairs',
    glasses: 'eyeglasses', grass: 'grass', bus: 'bus', clothes: 'clothes'};
  if (irr[w]) return irr[w];
  if (/ies$/.test(w) && w.length > 4) return w.slice(0, -3) + 'y';
  if (/(ches|shes|sses|xes)$/.test(w)) return w.slice(0, -2);
  if (/s$/.test(w) && !/ss$/.test(w) && w.length > 3) return w.slice(0, -1);
  return w;
}

function lookup(phrase: string): string | null {
  return BY_CLEAN.get(phrase) ?? (SYNONYMS[phrase] !== undefined ? SYNONYMS[phrase] : null);
}

/**
 * The table step: a PSG class for a free-text label, or null if it can't
 * tell (then ask Gemma, or leave it unknown).
 */
export function mapLabel(label: string): string | null {
  const raw = label.toLowerCase().replace(/[_]/g, ' ').replace(/[^a-z\s-]/g, ' ').replace(/\s+/g, ' ').trim();
  if (!raw) return null;
  const direct = lookup(raw);
  if (direct) return direct;
  const words = raw.split(' ').filter((w) => !ADJECTIVES.has(w)).map(singular);
  // Longest run of words first ("soccer ball" before "ball"), right-aligned: English head nouns come last.
  for (let n = Math.min(3, words.length); n >= 1; n--) {
    for (let i = words.length - n; i >= 0; i--) {
      const hit = lookup(words.slice(i, i + n).join(' '));
      if (hit) return hit;
    }
  }
  return null;
}

const CLEAN_LIST = [...new Set(PSG_CLASSES.map(cleanName))];

/** Text-only question for the labels the table missed. */
export function gemmaPrompt(labels: string[]): string {
  return 'Map each object label to the closest category in this list, or "none" if nothing fits.\n' +
      `Categories: ${CLEAN_LIST.join(', ')}.\n` +
      `Labels: ${JSON.stringify(labels)}\n` +
      'Answer only with a JSON object from each label to its category, e.g. {"goalie": "person"}.';
}

/**
 * Gemma's answer -> label -> PSG class (null = none / invalid). Only exact
 * category names count; everything else is unknown.
 */
export function parseGemmaReply(text: string, labels: string[]): Map<string, string | null> {
  const out = new Map<string, string | null>(labels.map((l) => [l, null]));
  const json = /\{[\s\S]*\}/.exec(text)?.[0] ?? /\{[\s\S]*$/.exec(text)?.[0];
  if (!json) return out;
  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(json) as Record<string, unknown>;
  } catch {
    // Tolerate a cut-off or sloppy object: "label": "category" pairs.
    obj = {};
    for (const m of json.matchAll(/"([^"]+)"\s*:\s*"([^"]*)"/g)) obj[m[1]] = m[2];
  }
  const byLower = new Map(labels.map((l) => [l.toLowerCase(), l]));
  for (const [k, v] of Object.entries(obj)) {
    const label = byLower.get(k.toLowerCase());
    if (!label || typeof v !== 'string') continue;
    out.set(label, BY_CLEAN.get(v.toLowerCase().trim()) ?? null);
  }
  return out;
}

/**
 * Label -> PSG class with a cache: the table first, Gemma (if given) for the
 * rest, answers checked.
 */
export class LabelMapper {
  private readonly cache = new Map<string, string | null>();
  private readonly asked = new Set<string>();

  /** Cached / table answer now ('' = unknown so far). */
  get(label: string): string {
    if (!label) return '';
    if (!this.cache.has(label)) {
      const t = mapLabel(label);
      if (t) this.cache.set(label, t);
      else return '';
    }
    return this.cache.get(label) ?? '';
  }

  /** Labels with no class yet that Gemma has not been asked about. */
  pending(labels: Iterable<string>): string[] {
    return [...new Set([...labels].filter((l) => l && !this.get(l) && !this.asked.has(l)))];
  }

  /** Asks Gemma about `labels` (one text-only request); returns the new answers. */
  async resolve(labels: string[], generate: (prompt: string) => Promise<string>): Promise<Map<string, string | null>> {
    const todo = this.pending(labels);
    if (!todo.length) return new Map();
    todo.forEach((l) => this.asked.add(l));
    const got = parseGemmaReply(await generate(gemmaPrompt(todo)), todo);
    for (const [l, c] of got) this.cache.set(l, c);
    return got;
  }
}
