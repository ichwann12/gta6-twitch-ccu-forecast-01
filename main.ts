// gta6-twitch-ccu-forecast-01 — forecast GTA VI's launch-day Twitch viewers.
//
// The program reads every case file <id>.in in the input directory (the grader
// runs one case at a time, with only that case's file there) and writes one
// forecast <id>.out per case to the output directory: 23 rows, hours 2..24 of
// the UTC release day, five quantiles (levels 0.05, 0.25, 0.5, 0.75, 0.95) of
// that hour's concurrent viewers in the game's Twitch category.
//
// The model — the release day's shape depends on how big its first hour was.
// A mid-size launch (hour 1 ~ 25k viewers) climbs to 3-4x its first hour by
// the evening, as time zones unlock and prime time arrives. A mega launch
// (hour 1 ~ 500k+) is front-loaded: it dips to about half its first hour by
// late morning and only recovers to roughly its first hour by the evening. A
// single pooled curve, and the house baseline's flat persistence, both miss
// this. So the forecast is local: for the case's hour-1 print p1, take the
// k = 9 past launches whose hour-1 print is closest to p1 in log space
// (among launches with a complete 24-hour release day and hour 1 >= 10,000
// viewers, where the hourly point is reliable), and for each hour take the
// median of those neighbours' log(viewers / hour-1) as the log-median ratio,
// with a spread from the neighbours' interquartile range. The quantiles are
// lognormal around that median. Nothing is hardcoded from any single launch:
// the program fits the neighbours from the comparables inside the case file
// it is given, so the final case's 42 comparables (including the four
// practice launches) are all used.
//
// Leave-one-out cross-validation over the kit's 38 past launches: the house
// baseline scores a mean pinball loss of 0.351 on the 28 launches with hour 1
// >= 10k and 0.226 on the six mega launches (hour 1 >= 300k); this model
// scores 0.293 and 0.127 on the same sets.
//
// It runs offline, with no network, under the case's time cap: everything it
// knows is in the case file. It is deterministic — no clock, no randomness —
// so two runs write the same bytes. A case that cannot be read still gets a
// valid output (persistence around 1 viewer), because an invalid case is
// scored as if every quantile equalled hour 1's print and forfeits any prize.
//
// Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
// stand in for /input and /output:
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** The hours this program forecasts. */
const HOURS = Array.from({ length: 23 }, (_, i) => i + 2);
/** Nearest neighbours to use, and the hour-1 floor for a neighbour to count. */
const K_NEIGHBOURS = 9;
const H1_FLOOR = 10000;
/** The log spread never collapses below this (log units). */
const SIGMA_FLOOR = 0.15;

interface KnownRow {
  hour: number;
  viewers: number;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: Array<{ from: string; viewers: number | null }>;
}

interface Case {
  levels?: number[];
  known?: KnownRow[];
  forecast?: Array<{ hour: number }>;
  comparables?: Comparable[];
}

/** A comparable's release day reduced to its 24 hourly counts. */
interface ReleaseCurve {
  name: string;
  h1: number;
  /** viewers by UTC hour of the release day, 1..24. */
  byHour: Map<number, number>;
}

/** The neighbours the fit chose for one hour: median log-ratio and log spread. */
interface HourFit {
  medianLogRatio: number;
  sigma: number;
}

/** Median of numbers (average of the two middle values when even). */
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 0;
  return n % 2 === 1 ? s[(n - 1) / 2]! : (s[n / 2 - 1]! + s[n / 2]!) / 2;
}

/** Linear-interpolation quantile at p in [0, 1]. */
function quantile(xs: number[], p: number): number {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) return 0;
  if (n === 1) return s[0]!;
  const i = p * (n - 1);
  const lo = Math.floor(i);
  const hi = Math.min(lo + 1, n - 1);
  const f = i - lo;
  return s[lo]! * (1 - f) + s[hi]! * f;
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

/**
 * Reduce one comparable to its release-day curve, or null when the day is not
 * complete: every UTC hour 1..24 needs a positive viewers count. The GTA VI
 * category's own pre-release history is not a release day and fails this test.
 */
function extractCurve(c: Comparable): ReleaseCurve | null {
  const dayStart = Date.parse(c.day_start);
  if (!Number.isFinite(dayStart)) return null;
  const byHour = new Map<number, number>();
  for (const pt of c.hourly ?? []) {
    if (pt.viewers === null || pt.viewers === undefined || pt.viewers <= 0) continue;
    const t = Date.parse(pt.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - dayStart) / 3600000) + 1;
    if (h >= 1 && h <= 24 && !byHour.has(h)) byHour.set(h, pt.viewers);
  }
  if (byHour.size !== 24) return null;
  return { name: c.name, h1: byHour.get(1)!, byHour };
}

/**
 * Fit the shape for an hour-1 print p1 from a pool of release-day curves.
 * Returns one fit per hour 2..24, or null when the pool holds no usable curve.
 */
function fitShape(curves: ReleaseCurve[], p1: number): Map<number, HourFit> | null {
  const usable = curves.filter((c) => c.h1 > 0);
  if (usable.length === 0) return null;
  // Nearest neighbours by |log h1 - log p1|, preferring launches whose hour 1
  // was big enough for the hourly point to be reliable; relax the floor (and
  // then the pool) only when too few qualify. Ties break by name so the
  // choice is deterministic.
  const logP1 = Math.log(Math.max(1, p1));
  let pool = usable.filter((c) => c.h1 >= H1_FLOOR);
  if (pool.length < K_NEIGHBOURS) pool = usable;
  const ranked = pool
    .map((c) => ({ c, d: Math.abs(Math.log(c.h1) - logP1) }))
    .sort((a, b) => a.d - b.d || (a.c.name < b.c.name ? -1 : a.c.name > b.c.name ? 1 : 0));
  const neighbours = ranked.slice(0, Math.min(K_NEIGHBOURS, ranked.length)).map((r) => r.c);
  if (neighbours.length === 0) return null;
  const fit = new Map<number, HourFit>();
  for (let h = 2; h <= 24; h++) {
    const ratios = neighbours.map((c) => Math.log(c.byHour.get(h)! / c.h1));
    const iqr = quantile(ratios, 0.75) - quantile(ratios, 0.25);
    fit.set(h, {
      medianLogRatio: median(ratios),
      sigma: Math.max(SIGMA_FLOOR, iqr / 1.349),
    });
  }
  return fit;
}

/** How many neighbours the fit used (0 when it fell back to persistence). */
function neighbourCount(curves: ReleaseCurve[], p1: number): number {
  const usable = curves.filter((c) => c.h1 > 0);
  const pool = usable.filter((c) => c.h1 >= H1_FLOOR);
  return Math.min(K_NEIGHBOURS, (pool.length >= K_NEIGHBOURS ? pool : usable).length);
}

/** The five quantiles for one hour from a fit (or the persistence fallback). */
function hourQuantiles(p1: number, fit: Map<number, HourFit> | null, h: number): number[] {
  const anchor = Math.max(1, p1);
  const f = fit?.get(h);
  const med = f === undefined ? 0 : f.medianLogRatio;
  // No fit at all: the house baseline's persistence spread, widening with horizon.
  const sig = f === undefined ? 0.15 * Math.sqrt(h - 1) : f.sigma;
  const qs = Z.map((z) => Math.round(anchor * Math.exp(med + z * sig) * 100) / 100);
  // Clamp to the contest's bounds and keep the row non-decreasing.
  let prev = 0.01;
  return qs.map((q) => {
    const clamped = Math.min(1e9, Math.max(0.01, q));
    const v = Math.max(prev, clamped);
    prev = v;
    return v;
  });
}

/** Forecast one parsed case into its 23 output rows. */
function forecastCase(c: Case): { hours: Array<{ hour: number; q: number[] }>; p1: number; neighbours: number; meanSigma: number } {
  const first = (c.known ?? []).find((k) => k.hour === 1);
  const p1 = first === undefined || !Number.isFinite(first.viewers) ? 1 : Math.max(1, first.viewers);
  const curves = (c.comparables ?? [])
    .map((comp) => extractCurve(comp))
    .filter((curve): curve is NonNullable<typeof curve> => curve !== null);
  const fit = fitShape(curves, p1);
  const hours = HOURS.map((h) => ({ hour: h, q: hourQuantiles(p1, fit, h) }));
  const sigmas = fit === null ? [0.15] : [...fit.values()].map((f) => f.sigma);
  return { hours, p1, neighbours: neighbourCount(curves, p1), meanSigma: mean(sigmas) };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let result: ReturnType<typeof forecastCase>;
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    if (c.levels !== undefined && (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j]))) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    result = forecastCase(c);
  } catch (err) {
    // A case that cannot be read still gets a valid output: persistence around 1.
    process.stderr.write(`gta6-twitch forecast: ${file} unreadable (${(err as Error).message}); writing persistence around 1\n`);
    result = {
      hours: HOURS.map((h) => ({ hour: h, q: hourQuantiles(1, null, h) })),
      p1: 1,
      neighbours: 0,
      meanSigma: 0.15,
    };
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours: result.hours })}\n`);
  const evening = result.hours.find((r) => r.hour === 19);
  const median19 = evening === undefined ? 0 : evening.q[2]!;
  process.stdout.write(
    `gta6-twitch forecast: ${id}: hour-1 print ${result.p1}, ${result.neighbours} neighbour launches, ` +
      `mean log spread ${result.meanSigma.toFixed(3)}, median hour-19 ratio ${(median19 / Math.max(1, result.p1)).toFixed(2)}x\n`,
  );
}
