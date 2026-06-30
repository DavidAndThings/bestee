export type JobStatus = "queued" | "running" | "completed" | "failed";

/** A SIC industry classification code and its title. */
export type SicCode = {
  sicCode: string;
  industryTitle: string;
};

/** A ticker symbol and its company name. */
export type SicTicker = {
  ticker: string;
  name: string | null;
};

/**
 * One `/search` match. `value` is what the field stores and the backend
 * resolves -- a ticker symbol for a security (stock or ETF), or a SIC industry
 * title. `label` is the display string (e.g. "Apple Inc (AAPL)"). `kind`
 * separates a single security from an industry that expands to many tickers;
 * `ticker`/`name` are set for securities so the dropdown can show both.
 */
export type SearchResult = {
  value: string;
  label: string;
  kind: "ticker" | "sic";
  ticker?: string | null;
  name?: string | null;
};

/** A submitted chart job, uniquely identified by a uuid. */
export type Job = {
  id: string;
  userId: string;
  /** The chart/schema this job ran. */
  schemaId: string;
  status: JobStatus;
  payload: Record<string, unknown>;
  /**
   * The deterministic, request-derived id under which the result is stored.
   * Known at submit time (returned by the API), so the `/results/{resultId}/
   * {aspect}` links can be built before the job finishes.
   */
  resultId?: string;
  createdAt: number;
  updatedAt: number;
};

/** One page of a user's jobs, mirroring the API's paginated `GET /jobs`. */
export type JobsPage = {
  jobs: Job[];
  /** Total jobs the user has (across all pages), for the page count. */
  total: number;
  offset: number;
  limit: number;
};

/** The full record of one job (its "log"), from `GET /jobs/{task_id}`. */
export type JobDetail = {
  taskId: string;
  schemaId: string;
  /** Raw Celery state (PENDING, STARTED, SUCCESS, FAILURE, ...). */
  state: string;
  status: JobStatus;
  payload: Record<string, unknown> | null;
  resultId: string | null;
  createdAt: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  elapsedSeconds: number | null;
  error: string | null;
  traceback: string | null;
};

/** One analysis-specific result aspect, as returned by `GET /results/{id}/{aspect}`. */
export type ResultTable = {
  resultId: string;
  aspect: string;
  columns: string[];
  rows: Record<string, unknown>[];
};
