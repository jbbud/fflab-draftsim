#!/usr/bin/env node

import { readFile, writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { resolve } from "node:path";
import { promisify } from "node:util";

const POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"];
const FLEX_POSITIONS = new Set(["RB", "WR", "TE"]);
const FLEX_REPLACEMENT_SHARE = { RB: 0.5, WR: 0.5, TE: 0 };
const BASELINE_WEIGHTS = {
  vor: 100,
  rank: 100,
  adp: 100,
  need: 20,
  dropoff: 0.6,
  handcuff: 1,
  stack: 1,
};
const ROSTER_SETTINGS = { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DEF: 1, BENCH: 6 };
const MAX_EXTRA = { QB: 1, RB: 4, WR: 4, TE: 1, K: 0, DEF: 0 };
const WEIGHT_LIMITS = {
  vor: [25, 200],
  rank: [25, 200],
  adp: [25, 200],
  need: [0, 80],
  dropoff: [0, 2.5],
  handcuff: [0, 8],
  stack: [0, 8],
};
const execFileAsync = promisify(execFile);

function number(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function integer(value, fallback = 0) {
  return Math.trunc(number(value, fallback));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function parseArgs(argv) {
  const args = {
    data: "",
    demo: false,
    out: "",
    candidates: 48,
    seeds: 8,
    survivors: 12,
    holdoutSeeds: 32,
    targetTeam: 0,
    seedBase: 20260721,
    logSpread: 0.75,
    traceOut: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2).replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    const value = argv[index + 1];
    if (value == null || value.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = value;
      index += 1;
    }
  }
  for (const key of ["candidates", "seeds", "survivors", "holdoutSeeds", "targetTeam", "seedBase"]) {
    args[key] = integer(args[key], args[key]);
  }
  args.logSpread = number(args.logSpread, 0.75);
  return args;
}

function rng(seed) {
  let state = integer(seed, 1) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function sortPlayers(players) {
  return players.slice().sort((a, b) => {
    const aRank = integer(a.rank);
    const bRank = integer(b.rank);
    if (aRank > 0 && bRank > 0) return aRank - bRank || number(b.projected_total_pts) - number(a.projected_total_pts);
    if (aRank > 0) return -1;
    if (bRank > 0) return 1;
    return number(b.projected_total_pts) - number(a.projected_total_pts);
  });
}

function normalizeBoardRanks(players) {
  sortPlayers(players).forEach((player, index) => {
    player.rank = index + 1;
    player.espn_rank = integer(player.espn_rank ?? player.rank);
  });
}

function demoPayload(numTeams = 14) {
  const positions = { QB: 32, RB: 72, WR: 84, TE: 36, K: 32, DEF: 32 };
  const base = { QB: 285, RB: 210, WR: 200, TE: 145, K: 125, DEF: 130 };
  const players = [];
  const weekly = [];
  for (const [position, count] of Object.entries(positions)) {
    for (let index = 1; index <= count; index += 1) {
      const playerId = position === "DEF" ? `DEF_${index}` : `${position}_${index}`;
      const total = Math.max(base[position] - index * (position === "QB" ? 5.2 : 2.6), 20);
      players.push({
        player_id: playerId,
        player_name: position === "DEF" ? `Team ${index} DEF` : `${position} Projection ${index}`,
        rank: players.length + 1,
        position,
        pro_team: position === "DEF" ? `T${index}` : "",
        pos_rank: index,
        adp: players.length + 1,
        projected_total_pts: Number(total.toFixed(2)),
      });
      for (let week = 1; week <= 17; week += 1) {
        const wave = (((index * 7 + week * 5) % 11) - 5) * 0.35;
        weekly.push({
          player_id: playerId,
          week,
          projected_points: Number(Math.max(total / 17 + wave, 0).toFixed(2)),
        });
      }
    }
  }
  return {
    players,
    weekly_projections: weekly,
    sessions: [{
      id: "active",
      num_teams: numTeams,
      team_names: Array.from({ length: numTeams }, (_, index) => `Team #${index + 1}`),
      roster_settings: ROSTER_SETTINGS,
      max_extra_per_position: MAX_EXTRA,
    }],
  };
}

async function syncEnvLeaguePayload() {
  const code = `
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))

from fflab.web import gui_config, load_default_env_files, payload_with_env_credentials
from fflab.projections import sync_projection_payload

load_default_env_files()
config = gui_config()
payload = {
    key: config[key]
    for key in ("league_id", "year", "week_start", "week_end")
    if key in config and str(config[key]).strip()
}
result = sync_projection_payload(payload_with_env_credentials(payload))
print(json.dumps(result))
`;
  const python = process.env.PYTHON || "python";
  const { stdout } = await execFileAsync(python, ["-c", code], {
    cwd: resolve("."),
    maxBuffer: 1024 * 1024 * 100,
  });
  return JSON.parse(stdout);
}

async function loadDataset(path, { demo = false } = {}) {
  let payload;
  let source;
  if (path) {
    source = resolve(path);
    payload = JSON.parse(await readFile(source, "utf8"));
  } else if (demo) {
    source = "built-in demo";
    payload = demoPayload();
  } else {
    source = "ESPN league from .env";
    payload = await syncEnvLeaguePayload();
  }
  const session = payload.sessions?.[0] || payload.session || {};
  const numTeams = Math.max(integer(session.num_teams, session.team_names?.length || 14), 2);
  const normalizedSession = {
    num_teams: numTeams,
    team_names: session.team_names || Array.from({ length: numTeams }, (_, index) => `Team #${index + 1}`),
    roster_settings: { ...ROSTER_SETTINGS, ...(session.roster_settings || {}) },
    max_extra_per_position: { ...MAX_EXTRA, ...(session.max_extra_per_position || {}) },
    draft_slots: session.draft_slots || payload.draft_slots || [],
  };
  const players = (payload.players || []).map((player) => ({
    ...player,
    player_id: String(player.player_id),
    rank: integer(player.rank),
    espn_rank: integer(player.espn_rank ?? player.rank),
    pos_rank: integer(player.pos_rank),
    adp: number(player.adp, -1),
    projected_total_pts: number(player.projected_total_pts ?? player.season_total_pts),
  }));
  normalizeBoardRanks(players);
  return {
    source,
    session: normalizedSession,
    players: sortPlayers(players),
    weekly: payload.weekly_projections || [],
  };
}

function totalRosterSlots(session) {
  return Object.values(session.roster_settings).reduce((sum, value) => sum + integer(value), 0);
}

function generateDraftSlots(session) {
  const slots = [];
  let overall = 1;
  for (let round = 1; round <= totalRosterSlots(session); round += 1) {
    const order = Array.from({ length: session.num_teams }, (_, index) => index);
    if (round % 2 === 0) order.reverse();
    for (let pickIndex = 0; pickIndex < order.length; pickIndex += 1) {
      slots.push({ overall, round, pick_in_round: pickIndex + 1, current_team: order[pickIndex], original_team: order[pickIndex] });
      overall += 1;
    }
  }
  return slots;
}

function baseDraftSlots(session) {
  const generated = generateDraftSlots(session);
  const synced = Array.isArray(session.draft_slots) ? session.draft_slots : [];
  if (!synced.length) return generated;
  const byOverall = new Map(synced.map((slot) => [integer(slot.overall), slot]));
  return generated.map((slot) => {
    const saved = byOverall.get(slot.overall);
    return saved ? { ...slot, current_team: clamp(integer(saved.current_team, slot.current_team), 0, session.num_teams - 1) } : slot;
  });
}

function rosterCounts(roster) {
  const counts = Object.fromEntries(POSITIONS.map((position) => [position, 0]));
  for (const player of roster) counts[player.position] = (counts[player.position] || 0) + 1;
  return counts;
}

function positionLimit(position, session) {
  return integer(session.roster_settings[position], 0) + integer(session.max_extra_per_position[position], 0);
}

function canAddPlayer(player, roster, drafted, session) {
  if (!player || drafted.has(String(player.player_id))) return false;
  if (roster.length >= totalRosterSlots(session)) return false;
  const limit = positionLimit(player.position, session);
  if (limit > 0 && rosterCounts(roster)[player.position] >= limit) return false;
  return true;
}

function positionSortedPlayers(players, position) {
  return players.filter((player) => player.position === position)
    .sort((a, b) => number(b.projected_total_pts) - number(a.projected_total_pts));
}

function replacementRankForPosition(position, session) {
  if (position === "K" || position === "DEF") return 1;
  const starters = integer(session.roster_settings[position], 0);
  const flexSlots = integer(session.roster_settings.FLEX, 0);
  const flexShare = FLEX_REPLACEMENT_SHARE[position] ?? 0;
  return Math.max(Math.ceil(session.num_teams * (starters + flexSlots * flexShare)), 1);
}

function replacementPointsByPosition(players, session) {
  const out = {};
  for (const position of [...new Set(players.map((player) => player.position))]) {
    const sorted = positionSortedPlayers(players, position);
    const idx = clamp(replacementRankForPosition(position, session) - 1, 0, Math.max(sorted.length - 1, 0));
    out[position] = number(sorted[idx]?.projected_total_pts, 0);
  }
  return out;
}

function vorScore(player, replacementByPosition) {
  return number(player.projected_total_pts) - number(replacementByPosition[player.position]);
}

function needTier(position, roster, session) {
  const counts = rosterCounts(roster);
  const starters = integer(session.roster_settings[position], 0);
  const openStarterSlots = Math.max(starters - integer(counts[position], 0), 0);
  const flexCapacity = integer(session.roster_settings.FLEX, 0);
  const flexWeight = { RB: 0.5, WR: 0.5, TE: 0.1 }[position] || 0;
  if (flexWeight <= 0 || flexCapacity <= 0) return openStarterSlots;
  const eligibleStarters = ["RB", "WR", "TE"].reduce((sum, pos) => sum + integer(session.roster_settings[pos], 0), 0);
  const eligibleDrafted = ["RB", "WR", "TE"].reduce((sum, pos) => sum + integer(counts[pos], 0), 0);
  const flexUsed = Math.max(eligibleDrafted - eligibleStarters, 0);
  return openStarterSlots + Math.max(flexCapacity - flexUsed, 0) * flexWeight;
}

function nextPickForTeam(slots, picks, teamIndex, afterOverall) {
  const picked = new Set(picks.map((pick) => integer(pick.overall)));
  return slots.find((slot) => integer(slot.current_team) === teamIndex && integer(slot.overall) > afterOverall && !picked.has(integer(slot.overall))) || null;
}

function dropoffScore(player, candidates, teamIndex, pick, slots, picks) {
  const sortedByPos = positionSortedPlayers(candidates, player.position);
  const currentIndex = sortedByPos.findIndex((row) => String(row.player_id) === String(player.player_id));
  if (currentIndex < 0) return 0;
  const nextPick = nextPickForTeam(slots, picks, teamIndex, pick.overall);
  if (!nextPick) return 0;
  const likelyGone = candidates
    .filter((candidate) => candidate.position === player.position)
    .filter((candidate) => number(candidate.adp, Infinity) > 0 && number(candidate.adp, Infinity) <= nextPick.overall)
    .length;
  const nextIndex = clamp(currentIndex + Math.max(likelyGone, 1), 0, sortedByPos.length - 1);
  return Math.max(number(player.projected_total_pts) - number(sortedByPos[nextIndex]?.projected_total_pts), 0);
}

function handcuffBonus(player, roster) {
  if (player.position !== "RB") return 0;
  const team = String(player.pro_team || "").toUpperCase();
  return roster.some((row) => row.position === "RB" && String(row.pro_team || "").toUpperCase() === team) ? 1 : 0;
}

function stackBonus(player, roster) {
  const team = String(player.pro_team || "").toUpperCase();
  if (!team) return 0;
  let bonus = 0;
  for (const mate of roster.filter((row) => String(row.pro_team || "").toUpperCase() === team && row.player_id !== player.player_id)) {
    const pair = [player.position, mate.position].sort().join("+");
    if (pair === "QB+WR") bonus += 8;
    else if (pair === "QB+TE") bonus += 6;
    else if (pair === "RB+WR") bonus += 4;
  }
  return bonus;
}

function rankScore(player, pool) {
  const ranks = [...new Set(pool.map((row) => integer(row.rank, 0)).filter((rank) => rank > 0))].sort((a, b) => a - b);
  const index = ranks.indexOf(integer(player.rank, 0));
  return index < 0 ? 0 : 1 / (index + 1);
}

function adpScore(player, pool) {
  const adps = [...new Set(pool.map((row) => number(row.adp, 0)).filter((adp) => adp > 0))].sort((a, b) => a - b);
  const index = adps.indexOf(number(player.adp, 0));
  return index < 0 ? 0 : 1 / (index + 1);
}

function quantile(sortedValues, percentile) {
  if (sortedValues.length === 0) return 0;
  if (sortedValues.length === 1) return sortedValues[0];
  const index = clamp(percentile, 0, 1) * (sortedValues.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

function signedPower(value, gamma) {
  return value === 0 ? 0 : Math.sign(value) * Math.pow(Math.abs(value), gamma);
}

function robustComponentNormalizer(values, gamma = 1) {
  const sorted = values.map((value) => number(value, NaN)).filter(Number.isFinite).sort((a, b) => a - b);
  if (sorted.length === 0) return () => 0;
  const median = quantile(sorted, 0.5);
  const spread = Math.max((quantile(sorted, 0.75) - quantile(sorted, 0.25)) / 1.349, 1e-6);
  return (value) => signedPower(clamp((number(value) - median) / spread / 2.5, -3, 3), gamma);
}

function positionTimingMultiplier(position, round) {
  if (position !== "K" && position !== "DEF") return 1;
  return Math.max(Math.pow(clamp(Math.max(integer(round, 1), 1) / 15, 0, 1), 6), 0.02);
}

function untimedValueShare(position) {
  return position === "K" || position === "DEF" ? 0 : 0.5;
}

function softmaxSample(items, scoreFn, random, temperature = 8) {
  const scores = items.map((item) => scoreFn(item));
  const maxScore = Math.max(...scores);
  const weights = scores.map((score) => Math.exp((score - maxScore) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (total <= 0) return items[0] || null;
  let roll = random() * total;
  for (let index = 0; index < items.length; index += 1) {
    roll -= weights[index];
    if (roll <= 0) return items[index];
  }
  return items.at(-1) || null;
}

function chooseBotPick({ teamIndex, pick, rosters, candidates, weights, replacementByPosition, slots, picks, random, session, trace }) {
  const candidatePool = sortPlayers(candidates).slice(0, 40);
  const rawScores = candidatePool.map((player) => ({
    player_id: String(player.player_id),
    vor: vorScore(player, replacementByPosition),
    rank: rankScore(player, candidatePool),
    adp: adpScore(player, candidatePool),
  }));
  const byId = new Map(rawScores.map((score) => [score.player_id, score]));
  const normalizeComponent = {
    vor: robustComponentNormalizer(rawScores.map((score) => score.vor), 1.15),
    rank: robustComponentNormalizer(rawScores.map((score) => score.rank), 1),
    adp: robustComponentNormalizer(rawScores.map((score) => score.adp), 1),
  };
  const roster = rosters[teamIndex] || [];
  const breakdowns = [];
  const selected = softmaxSample(candidatePool, (player) => {
    const raw = byId.get(String(player.player_id));
    const vorVal = normalizeComponent.vor(raw.vor);
    const rankVal = normalizeComponent.rank(raw.rank);
    const adpVal = normalizeComponent.adp(raw.adp);
    const timing = positionTimingMultiplier(player.position, pick.round);
    const coreShare = untimedValueShare(player.position);
    const timedShare = 1 - coreShare;
    const coreValue = vorVal * weights.vor * coreShare + rankVal * weights.rank * coreShare + adpVal * weights.adp * coreShare;
    const timedValue = (
      vorVal * weights.vor * timedShare
      + needTier(player.position, roster, session) * weights.need
      + dropoffScore(player, candidates, teamIndex, pick, slots, picks) * weights.dropoff
      + handcuffBonus(player, roster) * weights.handcuff
      + stackBonus(player, roster) * weights.stack
      + rankVal * weights.rank * timedShare
      + adpVal * weights.adp * timedShare
    );
    const total = coreValue + timedValue * timing;
    if (trace) {
      breakdowns.push({
        player_id: String(player.player_id),
        player_name: player.player_name,
        position: player.position,
        rank: player.rank,
        adp: player.adp,
        vor: raw.vor,
        vorVal,
        rankValRaw: raw.rank,
        rankVal,
        adpValRaw: raw.adp,
        adpVal,
        need: needTier(player.position, roster, session),
        dropoff: dropoffScore(player, candidates, teamIndex, pick, slots, picks),
        handcuff: handcuffBonus(player, roster),
        stack: stackBonus(player, roster),
        timing,
        coreValue,
        timedValue,
        score: total,
      });
    }
    return total;
  }, random);
  if (trace) {
    trace.push({
      seed: trace.seed,
      overall: pick.overall,
      round: pick.round,
      pick_in_round: pick.pick_in_round,
      team_index: teamIndex,
      selected_player_id: selected ? String(selected.player_id) : "",
      selected_player_name: selected?.player_name || "",
      candidates: breakdowns.sort((a, b) => b.score - a.score).slice(0, 12),
    });
  }
  return selected;
}

function runDraft(dataset, candidateWeights, seed, targetTeam, trace = null) {
  const random = rng(seed);
  const session = dataset.session;
  const slots = baseDraftSlots(session);
  const picks = [];
  const drafted = new Set();
  const rosters = Array.from({ length: session.num_teams }, () => []);
  const replacementByPosition = replacementPointsByPosition(dataset.players, session);
  for (const pick of slots) {
    const teamIndex = pick.current_team;
    const roster = rosters[teamIndex] || [];
    const candidates = dataset.players.filter((player) => canAddPlayer(player, roster, drafted, session));
    if (!candidates.length) continue;
    const weights = teamIndex === targetTeam ? candidateWeights : BASELINE_WEIGHTS;
    const player = chooseBotPick({ teamIndex, pick, rosters, candidates, weights, replacementByPosition, slots, picks, random, session, trace });
    if (!player) continue;
    drafted.add(String(player.player_id));
    roster.push(player);
    picks.push({ overall: pick.overall, team_index: teamIndex, player_id: String(player.player_id) });
  }
  return { rosters, picks };
}

function weeklyProjectionMap(weekly) {
  const map = new Map();
  for (const row of weekly) map.set(`${row.player_id}:${integer(row.week)}`, number(row.projected_points));
  return map;
}

function optimalLineup(roster, week, scores, session) {
  let remaining = roster.map((player) => ({ ...player, points: scores.get(`${player.player_id}:${week}`) || 0 }))
    .sort((a, b) => b.points - a.points);
  const starters = [];
  const take = (position, count) => {
    const matching = remaining.filter((player) => player.position === position).slice(0, count);
    starters.push(...matching);
    const taken = new Set(matching.map((player) => player.player_id));
    remaining = remaining.filter((player) => !taken.has(player.player_id));
  };
  for (const position of POSITIONS) take(position, integer(session.roster_settings[position], 0));
  const flex = remaining.filter((player) => FLEX_POSITIONS.has(player.position)).slice(0, integer(session.roster_settings.FLEX, 0));
  starters.push(...flex);
  return starters.reduce((sum, player) => sum + number(player.points), 0);
}

function rewardDraft(dataset, draft, targetTeam) {
  const weeks = [...new Set(dataset.weekly.map((row) => integer(row.week)).filter((week) => week > 0))].sort((a, b) => a - b);
  if (!weeks.length) {
    return draft.rosters[targetTeam].reduce((sum, player) => sum + number(player.projected_total_pts), 0);
  }
  const scores = weeklyProjectionMap(dataset.weekly);
  return weeks.reduce((sum, week) => sum + optimalLineup(draft.rosters[targetTeam] || [], week, scores, dataset.session), 0);
}

function evaluateWeights(dataset, weights, seeds, targetTeam) {
  const rewards = seeds.map((seed) => rewardDraft(dataset, runDraft(dataset, weights, seed, targetTeam), targetTeam));
  const mean = rewards.reduce((sum, value) => sum + value, 0) / rewards.length;
  const variance = rewards.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / rewards.length;
  return {
    weights,
    mean,
    variance,
    stdev: Math.sqrt(variance),
    worst: Math.min(...rewards),
    best: Math.max(...rewards),
    rewards,
  };
}

function sampleWeights(random, logSpread) {
  const weights = {};
  for (const [key, baseline] of Object.entries(BASELINE_WEIGHTS)) {
    const [min, max] = WEIGHT_LIMITS[key];
    const value = baseline * Math.exp((random() * 2 - 1) * logSpread);
    weights[key] = Number(clamp(value, min, max).toFixed(key === "dropoff" ? 3 : 2));
  }
  return weights;
}

function rankResults(results) {
  return results.slice().sort((a, b) => b.mean - a.mean || a.variance - b.variance || b.worst - a.worst);
}

function seedList(seedBase, count, offset = 0) {
  return Array.from({ length: count }, (_, index) => seedBase + offset + index * 9973);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const dataset = await loadDataset(args.data, { demo: Boolean(args.demo) });
  const searchRandom = rng(args.seedBase);
  const stageOneSeeds = seedList(args.seedBase, args.seeds);
  const stageTwoSeeds = seedList(args.seedBase, Math.max(args.seeds * 3, args.seeds + 1), 100000);
  const holdoutSeeds = seedList(args.seedBase, args.holdoutSeeds, 500000);
  const candidates = [
    BASELINE_WEIGHTS,
    ...Array.from({ length: args.candidates }, () => sampleWeights(searchRandom, args.logSpread)),
  ];
  const stageOne = rankResults(candidates.map((weights) => evaluateWeights(dataset, weights, stageOneSeeds, args.targetTeam)));
  const survivors = stageOne.slice(0, Math.max(1, Math.min(args.survivors, stageOne.length))).map((result) => result.weights);
  const stageTwo = rankResults(survivors.map((weights) => evaluateWeights(dataset, weights, stageTwoSeeds, args.targetTeam)));
  const baselineHoldout = evaluateWeights(dataset, BASELINE_WEIGHTS, holdoutSeeds, args.targetTeam);
  const bestHoldout = evaluateWeights(dataset, stageTwo[0].weights, holdoutSeeds, args.targetTeam);
  const report = {
    objective: "target team projected optimal-lineup season points",
    target_team: args.targetTeam,
    dataset: dataset.source,
    candidate_count: candidates.length,
    stage_one_seed_count: stageOneSeeds.length,
    stage_two_seed_count: stageTwoSeeds.length,
    holdout_seed_count: holdoutSeeds.length,
    baseline_holdout: baselineHoldout,
    best_holdout: bestHoldout,
    holdout_mean_delta: bestHoldout.mean - baselineHoldout.mean,
    top_stage_two: stageTwo.slice(0, 10),
  };
  if (args.traceOut) {
    const trace = [];
    trace.seed = holdoutSeeds[0];
    const tracedDraft = runDraft(dataset, bestHoldout.weights, holdoutSeeds[0], args.targetTeam, trace);
    await writeFile(resolve(args.traceOut), `${JSON.stringify({
      seed: holdoutSeeds[0],
      weights: bestHoldout.weights,
      reward: rewardDraft(dataset, tracedDraft, args.targetTeam),
      picks: trace,
    }, null, 2)}\n`, "utf8");
  }
  const text = JSON.stringify(report, null, 2);
  if (args.out) await writeFile(resolve(args.out), `${text}\n`, "utf8");
  console.log(text);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
