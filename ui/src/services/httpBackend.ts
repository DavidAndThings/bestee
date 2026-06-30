import type {
  Job,
  JobDetail,
  JobStatus,
  ResultTable,
  SearchResult,
  SicCode,
  SicTicker,
} from "../lib/types";
import type { SubmitJobInput, SubmitJobResult } from "./mockBackend";
import { getAuthToken } from "./auth";

/**
 * Real API client, interface-compatible with `mockBackend`. `jobsApi` selects
 * it when `VITE_API_BASE_URL` is set. Every request carries the Clerk bearer
 * token (see `auth.ts`).
 *
 * Jobs are persisted server-side at submit time (keyed by the Clerk user), so
 * `listJobs` reads the authoritative, cross-device history straight from
 * `GET /jobs` -- there is no browser-local index.
 */

// Tolerate stray surrounding quotes/backticks/whitespace from a hand-edited
// .env (e.g. a copy-pasted backtick) and drop any trailing slash, so
// `${API_BASE_URL}/path` is always a valid URL.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "")
  .trim()
  .replace(/^[`'"]+|[`'"]+$/g, "")
  .replace(/\/+$/, "");

/** UI schema id -> API `/tasks/{analysis}` path segment. */
const ANALYSIS_PATH: Record<string, string> = {
  "relative-rotation-graph": "rrg",
  "spectral-clustering": "clustering",
  "regime-detection": "regime",
  "fama-french": "fama-french",
};

/** API `analysis` value -> UI schema id (the result-id prefix `fama_french`,
 *  not the `fama-french` route segment). */
const SCHEMA_FOR_ANALYSIS: Record<string, string> = {
  rrg: "relative-rotation-graph",
  clustering: "spectral-clustering",
  regime: "regime-detection",
  fama_french: "fama-french",
};

/** One job as returned by `GET /jobs[/ {task_id}]` (`TaskStatus`). */
type ApiJobStatus = {
  task_id: string;
  state: string;
  result_id?: string | null;
  analysis?: string | null;
  payload?: Record<string, unknown> | null;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  elapsed_seconds?: number | null;
  error?: string | null;
  traceback?: string | null;
};

/** API `analysis` -> UI schema id (the result-id prefix, not the route). */
function schemaForAnalysis(analysis: string | null | undefined): string {
  if (!analysis) return "";
  return SCHEMA_FOR_ANALYSIS[analysis] ?? analysis;
}

/** The `GET /jobs` page envelope (`JobsPage`). */
type ApiJobsPage = {
  total: number;
  offset: number;
  limit: number;
  items: ApiJobStatus[];
};

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
  _userId: string,
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
  const data = (await response.json()) as {
    task_id: string;
    result_id: string;
  };
  // The API persists the job per user, so there is no browser-local bookkeeping.
  return {
    ok: true,
    requestId: data.task_id,
    resultId: data.result_id,
    receivedAt: Date.now(),
  };
}

export async function listJobs(userId: string): Promise<Job[]> {
  const response = await fetch(`${API_BASE_URL}/jobs?limit=100`, {
    headers: await authHeaders(),
  });
  if (!response.ok) {
    throw new Error(`Failed to load jobs (${response.status})`);
  }
  const data = (await response.json()) as ApiJobsPage;
  return data.items.map((item): Job => {
    const parsedCreated = item.created_at ? Date.parse(item.created_at) : NaN;
    const createdAt = Number.isNaN(parsedCreated) ? Date.now() : parsedCreated;
    const finished = item.finished_at ? Date.parse(item.finished_at) : NaN;
    return {
      id: item.task_id,
      userId,
      schemaId: schemaForAnalysis(item.analysis),
      status: toStatus(item.state),
      payload: item.payload ?? {},
      resultId: item.result_id ?? undefined,
      createdAt,
      updatedAt: Number.isNaN(finished) ? createdAt : finished,
    };
  });
}

export async function getJob(taskId: string): Promise<JobDetail> {
  const response = await fetch(
    `${API_BASE_URL}/jobs/${encodeURIComponent(taskId)}`,
    { headers: await authHeaders() },
  );
  if (!response.ok) {
    throw new Error(`Failed to load job (${response.status})`);
  }
  const item = (await response.json()) as ApiJobStatus;
  return {
    taskId: item.task_id,
    schemaId: schemaForAnalysis(item.analysis),
    state: item.state,
    status: toStatus(item.state),
    payload: item.payload ?? null,
    resultId: item.result_id ?? null,
    createdAt: item.created_at ?? null,
    startedAt: item.started_at ?? null,
    finishedAt: item.finished_at ?? null,
    elapsedSeconds: item.elapsed_seconds ?? null,
    error: item.error ?? null,
    traceback: item.traceback ?? null,
  };
}

/** One result aspect as returned by `GET /results/{id}/{aspect}` (`ResultTable`). */
type ApiResultTable = {
  result_id: string;
  aspect: string;
  columns: string[];
  rows: Record<string, unknown>[];
};

export async function getResultAspect(
  resultId: string,
  aspect: string,
): Promise<ResultTable> {
  const url = `${API_BASE_URL}/results/${encodeURIComponent(
    resultId,
  )}/${encodeURIComponent(aspect)}`;
  const response = await fetch(url, { headers: await authHeaders() });
  if (!response.ok) {
    // 404 typically means the job hasn't finished writing its result yet.
    const detail = response.status === 404 ? "not ready" : "failed";
    throw new Error(`Result table ${detail} (${response.status})`);
  }
  const data = (await response.json()) as ApiResultTable;
  return {
    resultId: data.result_id,
    aspect: data.aspect,
    columns: data.columns,
    rows: data.rows,
  };
}

/** One SIC row as returned by `GET /sic` (`SicCode`). */
type ApiSicCode = { sic_code: string; industry_title: string };

export async function listSicCodes(): Promise<SicCode[]> {
  const response = await fetch(`${API_BASE_URL}/sic`, {
    headers: await authHeaders(),
  });
  if (!response.ok) {
    console.error(
      `[bestee] /sic failed: ${response.status} ${response.statusText}`,
    );
    throw new Error(`Failed to load SIC codes (${response.status})`);
  }
  const data = (await response.json()) as { codes?: ApiSicCode[] };
  return (data.codes ?? []).map((code) => ({
    sicCode: code.sic_code,
    industryTitle: code.industry_title,
  }));
}

/** One ticker row as returned by `GET /sic/{code}/tickers` (`SicTicker`). */
type ApiSicTicker = { ticker: string; name: string | null };

export async function tickersForSic(sicCode: string): Promise<SicTicker[]> {
  const url = `${API_BASE_URL}/sic/${encodeURIComponent(sicCode)}/tickers`;
  const response = await fetch(url, { headers: await authHeaders() });
  if (!response.ok) {
    console.error(
      `[bestee] ${url} failed: ${response.status} ${response.statusText}`,
    );
    throw new Error(`Failed to load tickers (${response.status})`);
  }
  const data = (await response.json()) as { tickers?: ApiSicTicker[] };
  return (data.tickers ?? []).map((row) => ({
    ticker: row.ticker,
    name: row.name ?? null,
  }));
}

/** One `/search` match as returned by the API (snake/cased `SearchResult`). */
type ApiSearchResult = {
  value: string;
  label: string;
  kind: "ticker" | "sic";
  ticker?: string | null;
  name?: string | null;
};

export async function search(
  query: string,
  limit = 20,
): Promise<SearchResult[]> {
  const trimmed = query.trim();
  if (!trimmed) return [];
  const url = `${API_BASE_URL}/search?q=${encodeURIComponent(trimmed)}&limit=${limit}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  let response: Response;
  try {
    response = await fetch(url, {
      headers: await authHeaders(),
      signal: controller.signal,
    });
  } catch (cause) {
    // Network error, CORS rejection, or timeout. Surface it instead of
    // silently returning no matches; log the URL to aid debugging.
    console.error(`[bestee] search request failed: ${url}`, cause);
    throw new Error("Search request failed", { cause });
  } finally {
    clearTimeout(timeout);
  }
  if (!response.ok) {
    console.error(
      `[bestee] search failed: ${response.status} ${response.statusText} (${url})`,
    );
    throw new Error(`Search failed (${response.status})`);
  }
  const data = (await response.json()) as { results?: ApiSearchResult[] };
  return Array.isArray(data.results)
    ? data.results.map((r) => ({
        value: r.value,
        label: r.label,
        kind: r.kind,
        ticker: r.ticker ?? null,
        name: r.name ?? null,
      }))
    : [];
}
