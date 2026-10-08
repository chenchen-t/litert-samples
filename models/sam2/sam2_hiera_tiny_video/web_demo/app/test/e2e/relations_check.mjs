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

// Relations end-to-end check (real SAM 2 masks on WebGPU, system Chrome):
//   football sample -> player (2 clicks) + ball (click), labelled as Gemma
//   would -> track the clip -> relations scored on every tracked frame.
// Checks: relations exist on most frames, the player-ball relation is one of
// the expected predicates, scoring stays well under a frame, overlay drawn.
// Screenshots: test/e2e/screenshots/rel_*.png
//
//   node test/e2e/relations_check.mjs            (default: the PSG head as a
//       Tensor API model in the wasm pipeline on WebGPU, compared with the JS
//       head on every frame via ?relcheck=1)
//   REL=ts node test/e2e/relations_check.mjs     (the PSG head in JS, ?rel=ts)
//   REL=rules node test/e2e/relations_check.mjs  (the rule scorer)
import {mkdirSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {chromium} from 'playwright-core';
import {createServer} from 'vite';

const REL = ['rules', 'ts'].includes(process.env.REL) ? process.env.REL : 'tensorapi';
const app = resolve(import.meta.dirname, '../..');
const shots = resolve(app, 'test/e2e/screenshots');
mkdirSync(shots, {recursive: true});
const server = await createServer({root: app, logLevel: 'error', server: {port: 5178}});
await server.listen();
const browser = await chromium.launch({channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu']});
let failed = false;
const check = (ok, msg) => {
  console.log(`${ok ? '  ok ' : '  FAIL'} ${msg}`);
  if (!ok) failed = true;
};
try {
  const page = await browser.newPage({viewport: {width: 1440, height: 900}});
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error' || (m.type() === 'warning' && /relation|tensorapi/i.test(m.text()))) {
      console.log(`  [console] ${m.text()}`);
    }
  });
  await page.goto(`http://localhost:5178/${REL === 'tensorapi' ? '?relcheck=1' : `?rel=${REL}`}`);
  await page.waitForFunction(() => window.__sam2?.engine && window.__sam2?.clip && !window.__sam2.loading,
      null, {timeout: 300000});
  if (REL !== 'rules') {
    await page.waitForFunction(() => window.__relations.scorer() === 'learned', null, {timeout: 30000});
  }
  if (REL === 'tensorapi') {
    await page.waitForFunction(() => window.__relations.tensorApi().loaded, null, {timeout: 60000});
    check(true, 'Tensor API relation head authored + compiled in the wasm pipeline');
  }
  console.log('  scorer:', await page.evaluate(() => window.__relations.scorer()));
  const clip = await page.evaluate(() => ({w: window.__sam2.clip.width, h: window.__sam2.clip.height,
    n: window.__sam2.clip.frames.length}));
  console.log('  backend:', await page.textContent('#backendPill'));
  const click = async ([nx, ny]) => {
    const b = await page.locator('#view').boundingBox();
    const s = Math.min(b.width / clip.w, b.height / clip.h);
    await page.mouse.click(b.x + (b.width - clip.w * s) / 2 + nx * clip.w * s, b.y + (b.height - clip.h * s) / 2 + ny * clip.h * s);
    await page.waitForFunction(() => !window.__sam2.busy, null, {timeout: 60000});
  };
  // Same prompts as ui_check: the player (2 clicks) and the ball.
  await click([0.44, 0.28]);
  await click([0.45, 0.47]);
  await page.click('#addObjBtn');
  await click([0.484, 0.79]);
  // Labels as Gemma's Find would set them.
  await page.evaluate(() => {
    window.__sam2.objects[0].label = 'player';
    window.__sam2.objects[1].label = 'soccer ball';
  });
  const psg = await page.evaluate(() => window.__relations.psg().map((o) => `${o.label} -> ${o.psg}`));
  console.log('  PSG classes:', psg.join(', '));
  check(psg[0] === 'player -> person' && psg[1] === 'soccer ball -> sports ball', 'labels mapped to PSG classes');
  await page.click('#trackBtn');
  await page.waitForFunction(() => !window.__sam2.tracking && /Done|Stopped|failed/.test(
      document.getElementById('trackInfo').textContent), null, {timeout: 20 * 60 * 1000, polling: 1000});
  console.log(`  ${await page.textContent('#trackInfo')}`);

  if (REL === 'tensorapi') {
    const ta = await page.evaluate(() => {
      const r = window.__relations.tensorApi();
      const c = r.checks;
      const diffs = c.map((x) => x.maxDiff).sort((a, b) => a - b);
      return {frames: c.length, source: r.source, ms: r.ms,
        worst: diffs.at(-1) ?? NaN, median: diffs[Math.floor(diffs.length / 2)] ?? NaN,
        top1: c.reduce((a, x) => a + x.top1, 0), pairs: c.reduce((a, x) => a + x.pairs, 0)};
    });
    console.log(`  Tensor API vs TS head: ${ta.frames} frames, max |logit diff| median ${ta.median.toFixed(4)} ` +
        `worst ${ta.worst.toFixed(4)}, top-1 ${ta.top1}/${ta.pairs}, last run+readback ${ta.ms.toFixed(2)} ms`);
    check(ta.frames >= 0.9 * clip.n && ta.source === 'tensorapi', `Tensor API logits used on ${ta.frames}/${clip.n} frames`);
    // fp16 WebGPU vs fp32 JS; masks binarized at 0 on both sides, so a pixel on the edge can flip.
    check(ta.median < 0.1 && ta.top1 >= 0.97 * ta.pairs, 'Tensor API head matches the TS head');
  }

  const res = await page.evaluate(() => {
    const s = window.__sam2, R = window.__relations;
    const per = [];
    let maxMs = 0;
    for (let t = 0; t < s.clip.frames.length; t++) {
      const cur = R.at(t);
      per.push(cur && cur.t === t ? cur.rels.map((r) => ({...r, text: R.text(r)})) : null);
    }
    // Re-score one frame to time it in isolation.
    const cls = new Map(R.psg().map((o) => [o.id, o.psg]));
    const t0 = performance.now();
    for (let k = 0; k < 20; k++) {
      const m = s.results.get(100);
      if (m) R.engine.observe(100, 100 / s.clip.fps, s.objects.filter((o) => m.has(o.id))
          .map((o) => ({id: o.id, label: o.label, cls: cls.get(o.id), mask: m.get(o.id)})));
    }
    maxMs = (performance.now() - t0) / 20;
    return {per, maxMs};
  });
  const scored = res.per.filter((p) => p !== null);
  const withRel = scored.filter((p) => p.length);
  const counts = {};
  for (const p of withRel) for (const r of p) counts[r.predicate] = (counts[r.predicate] ?? 0) + 1;
  console.log('  predicate histogram (player-ball):', JSON.stringify(counts));
  check(scored.length >= 0.9 * clip.n, `relations scored on ${scored.length}/${clip.n} frames`);
  // Only 2 objects: on many frames the ball is far from the player and "no
  // relation" is right, so the bar is low; the predicate checks below matter more.
  check(withRel.length >= 0.15 * scored.length, `a relation shown on ${withRel.length}/${scored.length} scored frames`);
  const plausible = ['kicking', 'touching', 'near', 'beside', 'approaching', 'moving away', 'chasing', 'holding',
    'playing', 'playing with', 'about to hit', 'in front of', 'looking at', 'over', 'catching', 'throwing'];
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  check(top && plausible.includes(top[0]), `most frequent player-ball predicate is plausible: ${top?.[0]}`);
  check(res.maxMs < 5, `scoring 2 objects takes ${res.maxMs.toFixed(2)} ms per frame (JS, main thread)`);

  // Screenshots of frames with relations: first, middle, a "kicking" one if any.
  const pick = new Set();
  const idx = res.per.map((p, t) => (p?.length ? t : -1)).filter((t) => t >= 0);
  if (idx.length) { pick.add(idx[0]); pick.add(idx[Math.floor(idx.length / 2)]); }
  const kick = res.per.findIndex((p) => p?.some((r) => r.predicate === 'kicking'));
  if (kick >= 0) pick.add(kick);
  for (const t of pick) {
    await page.evaluate((t) => {
      window.__sam2.frame = t;
      document.getElementById('scrubber').value = String(t);
      document.getElementById('scrubber').dispatchEvent(new Event('input'));
    }, t);
    await page.waitForTimeout(400);
    const list = await page.textContent('#relList');
    console.log(`  frame ${t + 1}: ${list}`);
    await page.screenshot({path: `${shots}/rel_${REL}_f${t + 1}.png`});
  }
  check(pick.size > 0 && (await page.textContent('#relList')).includes('Relations'),
      'relation list shown under the video');

  // ---- predicate picker
  if (REL !== 'rules') {
    await page.click('#predBtn');
    await page.waitForTimeout(300);
    const chips = await page.locator('#relPredGroups input[type=checkbox]:not([disabled])').count();
    console.log(`  picker: ${await page.textContent('#relPredCount')}, ${chips} selectable`);
    check(chips === 53, `picker lists the 50 PSG + 3 motion predicates (${chips})`);
    await page.screenshot({path: `${shots}/rel_picker.png`});
    const used = () => page.evaluate(() => {
      const s = new Set();
      for (const f of window.__relations.log.list()) for (const r of f.relations) s.add(r.predicate);
      return [...s];
    });
    // Only "kicking": every logged relation is kicking (the head + calibration decide which frames).
    await page.locator('#relPredPresets button', {hasText: 'None'}).click();
    await page.locator('#relPredGroups label', {hasText: /^kicking$/}).locator('input').check();
    const onlyKick = await used();
    console.log(`  only "kicking": ${await page.evaluate(() => window.__relations.log.list().filter((f) => f.relations.length).length)} frames, predicates ${JSON.stringify(onlyKick)}`);
    check(onlyKick.every((p) => p === 'kicking'), 'single-predicate selection restricts the log');
    await page.locator('#relPredPresets button', {hasText: 'Spatial'}).click();
    const spatial = await used();
    console.log(`  Spatial preset: ${JSON.stringify(spatial)}`);
    check(spatial.every((p) => !['looking at', 'kicking', 'chasing', 'approaching'].includes(p)), 'Spatial preset excludes actions');
    await page.locator('#relPredPresets button', {hasText: 'All'}).click();
    const all = await used();
    check(all.includes('looking at'), `All restores every predicate (${JSON.stringify(all)})`);
    await page.click('#relPredClose');
  }

  // ---- relation log: frame idx -> relations, trace back, save
  await page.click('#logBtn');
  await page.waitForTimeout(400);
  const logInfo = await page.evaluate(() => {
    const L = window.__relations.log;
    const frames = L.list();
    return {n: frames.length, withRel: frames.filter((f) => f.relations.length).map((f) => f.frame),
      rows: document.querySelectorAll('#relLogList li[data-f]').length,
      count: document.getElementById('relLogCount').textContent};
  });
  check(logInfo.n === clip.n && logInfo.rows === logInfo.withRel.length && logInfo.rows > 0,
      `log has every tracked frame (${logInfo.n}); ${logInfo.rows} rows with relations (${logInfo.count})`);
  // Trace back: click a row in the middle; the video jumps to that frame and shows its relation.
  const target = logInfo.withRel[Math.floor(logInfo.withRel.length / 2)];
  await page.click(`#relLogList li[data-f="${target}"]`);
  await page.waitForTimeout(600);
  const after = await page.evaluate(() => ({frame: window.__sam2.frame, counter: document.getElementById('counter').textContent,
    rel: document.getElementById('relList').textContent,
    cur: document.querySelector('#relLogList li.cur')?.getAttribute('data-f')}));
  check(after.frame === target && after.cur === String(target) && after.rel.includes('Relations'),
      `click frame ${target} in the log -> video at frame idx ${after.frame} (${after.counter}), row highlighted, "${after.rel}"`);
  await page.screenshot({path: `${shots}/rel_log_frames.png`});
  await page.click('#relLogView button[data-v="events"]');
  await page.waitForTimeout(300);
  const events = await page.$$eval('#relLogList li[data-f]', (ls) => ls.map((l) => l.textContent));
  console.log('  events:', events.slice(0, 6).join(' | '));
  check(events.length > 0 && events.length < logInfo.rows, `Events view condenses ${logInfo.rows} rows into ${events.length} spans`);
  await page.screenshot({path: `${shots}/rel_log_events.png`});
  const [dl] = await Promise.all([page.waitForEvent('download'), page.click('#relLogJson')]);
  const jsonPath = resolve(shots, dl.suggestedFilename());
  await dl.saveAs(jsonPath);
  const saved = JSON.parse(readFileSync(jsonPath, 'utf8'));
  check(saved.meta.source === 'Football (sample video)' && saved.frames.length === clip.n && saved.spans.length === events.length,
      `JSON saved (${dl.suggestedFilename()}): ${saved.frames.length} frames, ${saved.spans.length} spans`);
  const [dl2] = await Promise.all([page.waitForEvent('download'), page.click('#relLogCsv')]);
  const csvPath = resolve(shots, dl2.suggestedFilename());
  await dl2.saveAs(csvPath);
  const csv = readFileSync(csvPath, 'utf8').trim().split('\n');
  console.log('  csv:', csv.slice(0, 3).join(' / '));
  check(csv[0].startsWith('frame,time_s') && csv.length > 1, `CSV saved: ${csv.length - 1} rows`);
} catch (e) {
  console.log('  FAIL', e);
  failed = true;
} finally {
  await browser.close();
  await server.close();
}
console.log(failed ? 'RELATIONS CHECK FAILED' : 'RELATIONS CHECK PASSED');
process.exit(failed ? 1 : 0);
