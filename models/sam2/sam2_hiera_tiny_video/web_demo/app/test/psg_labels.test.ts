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

import {describe, expect, it, vi} from 'vitest';
import {cleanName, gemmaPrompt, LabelMapper, mapLabel, parseGemmaReply, PSG_CLASSES} from '../src/psg_labels';

describe('mapLabel (table)', () => {
  it.each([
    ['player', 'person'], ['players', 'person'], ['soccer player', 'person'], ['goalkeeper', 'person'],
    ['Referee', 'person'], ['people', 'person'], ['a man', 'person'], ['player in red', 'person'],
    ['soccer ball', 'sports ball'], ['football', 'sports ball'], ['the ball', 'sports ball'],
    ['white soccer ball', 'sports ball'], ['pitch', 'playingfield'], ['football field', 'playingfield'],
    ['grass', 'grass-merged'], ['sky', 'sky-other-merged'], ['red car', 'car'], ['cars', 'car'],
    ['puppy', 'dog'], ['bike', 'bicycle'], ['phone', 'cell phone'], ['dining table', 'dining table'],
    ['brick wall', 'wall-brick'], ['wall', 'wall-other-merged'], ['orange', 'orange'], ['tv', 'tv'],
    ['sports ball', 'sports ball'], ['grass-merged', 'grass-merged'],
  ])('%s -> %s', (label, cls) => {
    expect(mapLabel(label)).toBe(cls);
  });

  it.each(['corner flag', 'glasses', 'drone', '', '   '])('%s -> null', (label) => {
    expect(mapLabel(label)).toBeNull();
  });

  it('every class maps to itself, raw and cleaned', () => {
    for (const c of PSG_CLASSES) {
      expect(mapLabel(c)).toBe(c);
      expect(mapLabel(cleanName(c))).toBe(c);
    }
  });
});

describe('Gemma fallback', () => {
  it('prompt lists clean names and the labels', () => {
    const p = gemmaPrompt(['goalie gloves', 'corner flag']);
    expect(p).toContain('"corner flag"');
    expect(p).toContain('playingfield');
    expect(p).toContain('grass,');
    expect(p).not.toContain('grass-merged');
  });

  it('parses and validates replies', () => {
    const got = parseGemmaReply('```json\n{"corner flag": "banner", "Drone": "airplane", "x": "spaceship"}\n```',
        ['corner flag', 'drone', 'x', 'missing']);
    expect(got.get('corner flag')).toBe('banner');
    expect(got.get('drone')).toBe('airplane');
    expect(got.get('x')).toBeNull();        // not a PSG class
    expect(got.get('missing')).toBeNull();  // not answered
  });

  it('tolerates a sloppy object', () => {
    const got = parseGemmaReply('{"drone": "airplane", "kite thing": "kite",', ['drone', 'kite thing']);
    expect(got.get('drone')).toBe('airplane');
    expect(got.get('kite thing')).toBe('kite');
  });

  it('LabelMapper asks Gemma once, only for unmapped labels', async () => {
    const m = new LabelMapper();
    expect(m.get('player')).toBe('person');
    expect(m.get('drone')).toBe('');
    const gen = vi.fn(async (prompt: string) => {
      expect(prompt).toContain('"drone"');
      expect(prompt).not.toContain('"player"');
      return '{"drone": "airplane"}';
    });
    await m.resolve(['player', 'drone', 'drone'], gen);
    expect(gen).toHaveBeenCalledTimes(1);
    expect(m.get('drone')).toBe('airplane');
    await m.resolve(['drone'], gen);
    expect(gen).toHaveBeenCalledTimes(1);
    expect(m.pending(['drone', 'player'])).toEqual([]);
  });
});
