import * as httpBackend from "./httpBackend";
import * as mockBackend from "./mockBackend";
import type { SubmitJobInput, SubmitJobResult } from "./mockBackend";
import type { Job } from "../lib/types";

export type { SubmitJobInput, SubmitJobResult } from "./mockBackend";

/** The data-layer contract shared by the mock and the real HTTP backend. */
type JobsBackend = {
  listJobs(userId: string): Promise<Job[]>;
  submitJob(userId: string, input: SubmitJobInput): Promise<SubmitJobResult>;
  search(query: string, limit?: number): Promise<string[]>;
};

// Use the real API when VITE_API_BASE_URL is configured; otherwise fall back to
// the localStorage-backed mock so the UI runs standalone in development.
const backend: JobsBackend = import.meta.env.VITE_API_BASE_URL
  ? httpBackend
  : mockBackend;

/**
 * Jobs data layer and single integration point. Swaps between the mock and the
 * live API based on configuration, without the pages or hooks needing to know.
 */
export const jobsApi = {
  listJobs(userId: string): Promise<Job[]> {
    return backend.listJobs(userId);
  },
  submitJob(userId: string, input: SubmitJobInput): Promise<SubmitJobResult> {
    return backend.submitJob(userId, input);
  },
  /** Search the term catalog (company names + SIC industry titles). */
  search(query: string, limit?: number): Promise<string[]> {
    return backend.search(query, limit);
  },
};
