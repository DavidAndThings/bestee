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

/** One analysis-specific result aspect, as returned by `GET /results/{id}/{aspect}`. */
export type ResultTable = {
  resultId: string;
  aspect: string;
  columns: string[];
  rows: Record<string, unknown>[];
};
