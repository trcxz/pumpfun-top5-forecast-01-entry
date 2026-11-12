// pumpfun-top5-forecast-01 entry: a momentum-weighted rank model plus a
// saturating market-cap growth model for the size bins.
//
// Order: each place k gets probability mass over the coins whose current rank
// is near k, weighted by their log-cap momentum (log10 cap gained per mark) and
// rank velocity (places climbed per mark). A fixed share of every place is left
// unnamed so a coin launched after the history ends can still take it, as the
// rules say such a coin may.
//
// Sizes: each place's cap is centred on the current cap at that rank, moved up
// by a growth term that saturates with the forecast horizon (the pool of coins
// matures, so early log-cap growth does not extrapolate linearly), with a wide
// sd because a cap moves a long way in the hours between the history and the
// reads.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  symbol?: string;
  created_at?: string;
}
interface Mark {
  at?: string;
  ok: boolean;
  rows: Row[] | null;
}
interface Task {
  places: number;
  cadence_minutes?: number;
  reads?: string[];
  size: { edges: number[] };
  output: string;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const marks: Mark[] = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as Mark);
const ok = marks.filter((m) => m.ok === true && Array.isArray(m.rows) && m.rows.length > 0);
if (ok.length === 0) throw new Error('the history holds no ok mark with rows');

const last = ok[ok.length - 1]!;
const chart = [...last.rows!].sort((a, b) => a.rank - b.rank);
const places = task.places;
const edges = task.size.edges;
const MICRO = 1_000_000;

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
const logCap = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

/** Per mint: its rank and log cap at each mark it appears in. */
const hist = new Map<string, { i: number; rank: number; lc: number }[]>();
ok.forEach((m, i) => {
  for (const r of m.rows!) {
    const a = hist.get(r.mint) ?? [];
    a.push({ i, rank: r.rank, lc: logCap(r.market_cap_lamports) });
    hist.set(r.mint, a);
  }
});

/** log10 cap gained per mark and places climbed per mark, over the coin's span. */
function momentum(mint: string): number {
  const h = hist.get(mint);
  if (h === undefined || h.length < 2) return 0;
  const first = h[0]!;
  const lastH = h[h.length - 1]!;
  const d = lastH.i - first.i;
  if (d <= 0) return 0;
  const dlc = (lastH.lc - first.lc) / d;
  const dr = (first.rank - lastH.rank) / d;
  return 0.7 * dlc + 0.03 * dr;
}

/** erfc(x), Numerical Recipes' erfcc (fractional error under 1.2e-7). */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}
/** P(X >= x) for X normal with mean and sd. */
const above = (x: number, mean: number, sd: number): number =>
  x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2));

// ---------------------------------------------------------------- the order model
/** How wide a rank's reach is: a coin this many ranks away is still plausible. */
const SIGMA_RANK = 5;
/** How much one unit of momentum lifts a coin's log score. */
const W_MOMENTUM = 2.0;
/** The share of each place left unnamed, for a coin launched after the history ends. */
const UNNAMED = 0.35;
/** Candidates: the top of the last chart, where a top-5 coin at a read must come from. */
const CANDIDATES = 40;

const candidates = chart.slice(0, CANDIDATES);
const mom = new Map<string, number>();
for (const c of candidates) mom.set(c.mint, momentum(c.mint));

/** A coin's log score for one place: its rank's distance plus its momentum. */
function logScore(rank: number, m: number, place: number): number {
  const d = (rank - place) / SIGMA_RANK;
  return -0.5 * d * d + W_MOMENTUM * m;
}

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let place = 1; place <= places; place++) {
  const scored = candidates.map((c) => ({ mint: c.mint, s: logScore(c.rank, mom.get(c.mint) ?? 0, place) }));
  const hi = Math.max(...scored.map((x) => x.s));
  let z = 0;
  const w = scored.map((x) => {
    const v = Math.exp(x.s - hi);
    z += v;
    return v;
  });
  // Keep only the coins that carry real mass, so the file stays small and the
  // named mass stays where it is useful.
  const mass = scored.map((x, i) => ({ mint: x.mint, p: (w[i]! / z) * (1 - UNNAMED) })).filter((x) => x.p >= 0.004);
  let sum = 0;
  for (const x of mass) {
    const u = Math.floor(x.p * MICRO);
    if (u > 0) {
      probabilities.push({ mint: x.mint, place, p: u / MICRO });
      sum += u / MICRO;
    }
  }
  void sum;
}

// ---------------------------------------------------------------- the size model
/** The most log10 growth the model will add, however long the horizon. */
const G_MAX = 0.9;
/** The horizon, in marks, at which that growth is ~63% reached. */
const GROWTH_TAU = 8;

const cadence = task.cadence_minutes ?? 10;
const lastAt = last.at !== undefined ? Date.parse(last.at) : NaN;
const reads = (task.reads ?? []).map((r) => Date.parse(r));
const meanRead = reads.length > 0 ? reads.reduce((a, b) => a + b, 0) / reads.length : NaN;
const horizonMarks =
  Number.isFinite(lastAt) && Number.isFinite(meanRead) ? Math.max(0, (meanRead - lastAt) / 60000 / cadence) : GROWTH_TAU;
const growth = G_MAX * (1 - Math.exp(-horizonMarks / GROWTH_TAU));

const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let place = 1; place <= places; place++) {
  const row = chart[place - 1];
  if (row === undefined) continue;
  // Later places have more room to grow, and more uncertainty.
  const mean = logCap(row.market_cap_lamports) + growth * (1 + 0.05 * (place - 1));
  const sd = 0.5 + 0.025 * (place - 1);
  for (let bin = 0; bin <= edges.length; bin++) {
    const lo = bin === 0 ? -Infinity : edges[bin - 1]!;
    const hi = bin === edges.length ? Infinity : edges[bin]!;
    const u = Math.floor((above(lo, mean, sd) - above(hi, mean, sd)) * MICRO);
    if (u > 0) sizes.push({ place, bin, p: u / MICRO });
  }
}

// ---------------------------------------------------------------- top5, the tie-break
/** Each candidate's expected place, from the mass it holds over the five places. */
const expected = new Map<string, { mass: number; place: number }>();
for (const r of probabilities) {
  const e = expected.get(r.mint) ?? { mass: 0, place: 0 };
  e.mass += r.p;
  e.place += r.p * r.place;
  expected.set(r.mint, e);
}
const top5 = [...expected.entries()]
  .filter(([, e]) => e.mass > 0)
  .sort((a, b) => a[1].place / a[1].mass - b[1].place / b[1].mass || b[1].mass - a[1].mass)
  .slice(0, places)
  .map(([mint]) => mint);
// Fill the tail from the last chart if the model named fewer than five.
for (const r of chart) {
  if (top5.length >= places) break;
  if (!top5.includes(r.mint)) top5.push(r.mint);
}

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
