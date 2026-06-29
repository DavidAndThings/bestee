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
  createdAt: number;
  updatedAt: number;
};
