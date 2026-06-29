import type { Job, JobStatus } from "../lib/types";
import type { SubmitJobInput, SubmitJobResult } from "./mockBackend";
import { getAuthToken } from "./auth";

/**
 * Real API client, interface-compatible with `mockBackend`. `jobsApi` selects
 * it when `VITE_API_BASE_URL` is set. Every request carries the Clerk bearer
 * token (see `auth.ts`).
 *
 * The Celery result backend can't return a job's original schema/payload, so
 * each submission is recorded in a per-user localStorage index; `listJobs`
 * reconciles that index with the authoritative state polled from the API.
 */

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

/** UI schema id -> API `/tasks/{analysis}` path segment. */
const ANALYSIS_PATH: Record<string, string> = {
  "relative-rotation-graph": "rrg",
  "spectral-clustering": "clustering",
  "regime-detection": "regime",
  "fama-french": "fama-french",
};

const REMOTE_JOBS_PREFIX = "bestee:remote-jobs:";

type StoredJob = {
  taskId: string;
  schemaId: string;
  payload: Record<string, unknown>;
  createdAt: number;
};

/** A single job's state as returned by `GET /jobs/{task_id}` (`TaskStatus`). */
type TaskStatus = {
  task_id: string;
  state: string;
  finished_at?: string | null;
};

function indexKey(userId: string): string {
  return `${REMOTE_JOBS_PREFIX}${userId}`;
}

function readIndex(userId: string): StoredJob[] {
  try {
    const raw = localStorage.getItem(indexKey(userId));
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? (parsed as StoredJob[]) : [];
  } catch {
    return [];
  }
}

function writeIndex(userId: string, jobs: StoredJob[]): void {
  try {
    localStorage.setItem(indexKey(userId), JSON.stringify(jobs));
  } catch {
    // Storage may be unavailable (private mode, quota) — ignore.
  }
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** Map a Celery task state to the UI's coarse status. */
function toStatus(state: string): JobStatus {
  switch (state) {
    case "SUCCESS":
      return "completed";
    case "FAILURE":
    case "REVOKED":
      return "failed";
    case "STARTED":
    case "RETRY":
      return "running";
    default:
      return "queued"; // PENDING, RECEIVED, or unknown
  }
}

export async function submitJob(
  userId: string,
  input: SubmitJobInput,
): Promise<SubmitJobResult> {
  const analysis = ANALYSIS_PATH[input.schemaId];
  if (!analysis) {
    throw new Error(`Unknown analysis: ${input.schemaId}`);
  }
  const response = await fetch(`${API_BASE_URL}/tasks/${analysis}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await authHeaders()) },
    body: JSON.stringify(input.payload),
  });
  if (!response.ok) {
    throw new Error(`Submit failed (${response.status})`);
  }
  const data = (await response.json()) as { task_id: string };
  const now = Date.now();
  const jobs = readIndex(userId);
  jobs.push({
    taskId: data.task_id,
    schemaId: input.schemaId,
    payload: input.payload,
    createdAt: now,
  });
  writeIndex(userId, jobs);
  return { ok: true, requestId: data.task_id, receivedAt: now };
}

export async function listJobs(userId: string): Promise<Job[]> {
  const stored = readIndex(userId)
    .slice()
    .sort((a, b) => b.createdAt - a.createdAt);
  const headers = await authHeaders();
  return Promise.all(
    stored.map(async (record): Promise<Job> => {
      let status: JobStatus = "queued";
      let updatedAt = record.createdAt;
      try {
        const response = await fetch(`${API_BASE_URL}/jobs/${record.taskId}`, {
          headers,
        });
        if (response.ok) {
          const meta = (await response.json()) as TaskStatus;
          status = toStatus(meta.state);
          if (meta.finished_at) {
            const finished = Date.parse(meta.finished_at);
            if (!Number.isNaN(finished)) updatedAt = finished;
          }
        }
      } catch {
        // Network hiccup — leave this job as queued for now.
      }
      return {
        id: record.taskId,
        userId,
        schemaId: record.schemaId,
        status,
        payload: record.payload,
        createdAt: record.createdAt,
        updatedAt,
      };
    }),
  );
}

export async function search(query: string, limit = 20): Promise<string[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  try {
    const url = `${API_BASE_URL}/search?q=${encodeURIComponent(trimmed)}&limit=${limit}`;
    const response = await fetch(url, { headers: await authHeaders() });
    if (!response.ok) return [];
    const data = (await response.json()) as { results?: string[] };
    return Array.isArray(data.results) ? data.results : [];
  } catch {
    return [];
  }
}
