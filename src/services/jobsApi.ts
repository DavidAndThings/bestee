import * as mockBackend from "./mockBackend";
import type { Job } from "../lib/types";

/**
 * Jobs data layer. Delegates to the localStorage-backed mock today; point this
 * at a real API later without touching the page or hook.
 */
export const jobsApi = {
  listJobs(userId: string): Promise<Job[]> {
    return mockBackend.listJobs(userId);
  },
};
