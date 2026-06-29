import * as mockBackend from "./mockBackend";
import type { Job } from "../lib/types";

export type { SubmitJobInput, SubmitJobResult } from "./mockBackend";

/**
 * Jobs data layer. Delegates to the localStorage-backed mock today; point this
 * at a real API later without touching the page or hook.
 */
export const jobsApi = {
  listJobs(userId: string): Promise<Job[]> {
    return mockBackend.listJobs(userId);
  },
  submitJob(
    userId: string,
    input: mockBackend.SubmitJobInput,
  ): Promise<mockBackend.SubmitJobResult> {
    return mockBackend.submitJob(userId, input);
  },
  /** Search the term catalog (company names + SIC industry titles). */
  search(query: string, limit?: number): Promise<string[]> {
    return mockBackend.search(query, limit);
  },
};
