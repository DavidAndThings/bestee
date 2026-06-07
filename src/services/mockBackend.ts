import type { Conversation, Job } from "../lib/types";

/**
 * A stand-in for a real backend. The user's single conversation is persisted to
 * localStorage, namespaced per Clerk user id, and every operation simulates
 * network latency. Swap this file's internals for real HTTP calls without
 * touching the rest of the app (see `chatApi.ts`, the single integration point).
 */

const STORAGE_PREFIX = "bestee:chat:";
const JOBS_PREFIX = "bestee:jobs:";

function delay<T>(value: T, ms = 250): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

function storageKey(userId: string): string {
  return `${STORAGE_PREFIX}${userId}`;
}

export function loadConversation(userId: string): Promise<Conversation | null> {
  try {
    const raw = localStorage.getItem(storageKey(userId));
    if (!raw) return delay(null);
    const parsed = JSON.parse(raw) as unknown;
    const valid =
      parsed &&
      typeof parsed === "object" &&
      Array.isArray((parsed as Conversation).messages);
    return delay(valid ? (parsed as Conversation) : null);
  } catch {
    return delay(null);
  }
}

export function saveConversation(
  userId: string,
  conversation: Conversation,
): Promise<Conversation> {
  localStorage.setItem(storageKey(userId), JSON.stringify(conversation));
  return delay(conversation, 120);
}

// ── Jobs ───────────────────────────────────────────────────────────────────

function jobsKey(userId: string): string {
  return `${JOBS_PREFIX}${userId}`;
}

function readJobs(userId: string): Job[] {
  try {
    const raw = localStorage.getItem(jobsKey(userId));
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as Job[]) : [];
  } catch {
    return [];
  }
}

function writeJobs(userId: string, jobs: Job[]): void {
  localStorage.setItem(jobsKey(userId), JSON.stringify(jobs));
}

/** Mock data: a handful of previously submitted jobs across tools/statuses. */
function createSeedJobs(userId: string): Job[] {
  const now = Date.now();
  const minute = 60_000;
  const hour = 60 * minute;
  const day = 24 * hour;

  const make = (
    schemaId: string,
    status: Job["status"],
    ago: number,
    payload: Record<string, unknown>,
  ): Job => ({
    id: crypto.randomUUID(),
    userId,
    schemaId,
    status,
    payload,
    createdAt: now - ago,
    updatedAt: now - ago,
  });

  return [
    make("relative-rotation-graph", "completed", 2 * day, {
      securities: ["AAPL", "MSFT", "GOOG"],
      lookback_window: 14,
      lookback_period: "1w",
      smoothing_method: "z-score",
      smoothing_window: 5,
    }),
    make("portfolio-backtest", "completed", 5 * hour, {
      portfolio_name: "Tech Core",
      securities: ["AAPL", "NVDA"],
      initial_capital: 100000,
      rebalance_frequency: "monthly",
      lookback_window: 60,
    }),
    make("relative-rotation-graph", "failed", 1 * day, {
      securities: ["TSLA", "RIVN"],
      lookback_window: 30,
      lookback_period: "1d",
      smoothing_method: "double-ema",
      smoothing_window: 10,
    }),
    make("portfolio-backtest", "running", 20 * minute, {
      portfolio_name: "Dividend Mix",
      securities: ["JNJ", "KO", "PG"],
      initial_capital: 50000,
      rebalance_frequency: "quarterly",
      lookback_window: 90,
    }),
    make("relative-rotation-graph", "queued", 3 * minute, {
      securities: ["SPY", "QQQ", "IWM"],
      lookback_window: 21,
      lookback_period: "1w",
      smoothing_method: "z-score",
      smoothing_window: 7,
    }),
  ];
}

function ensureSeeded(userId: string): Job[] {
  const existing = readJobs(userId);
  if (existing.length > 0) return existing;
  const seeded = createSeedJobs(userId);
  writeJobs(userId, seeded);
  return seeded;
}

export function listJobs(userId: string): Promise<Job[]> {
  const jobs = ensureSeeded(userId)
    .slice()
    .sort((a, b) => b.createdAt - a.createdAt);
  return delay(jobs);
}

export type SubmitInput = {
  schemaId: string;
  payload: Record<string, unknown>;
};

export type SubmitResult = {
  ok: boolean;
  requestId: string;
  receivedAt: number;
};

/**
 * Pretends to dispatch a completed tool call's payload to a backend, and
 * records it as a tracked job (uniquely identified by `requestId`).
 */
export function submitPayload(
  userId: string,
  input: SubmitInput,
): Promise<SubmitResult> {
  const requestId = crypto.randomUUID();
  const now = Date.now();
  const job: Job = {
    id: requestId,
    userId,
    schemaId: input.schemaId,
    status: "queued",
    payload: input.payload,
    createdAt: now,
    updatedAt: now,
  };
  const jobs = ensureSeeded(userId);
  jobs.push(job);
  writeJobs(userId, jobs);

  console.info("[mockBackend] submit", {
    userId,
    schemaId: input.schemaId,
    requestId,
    payload: input.payload,
  });
  return delay({ ok: true, requestId, receivedAt: now }, 900);
}
