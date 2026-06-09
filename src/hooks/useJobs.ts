import { useCallback, useEffect, useState } from "react";
import type { Job } from "../lib/types";
import { jobsApi } from "../services/jobsApi";

export type JobsState = {
  jobs: Job[];
  loading: boolean;
  refresh: () => void;
};

/** Loads the signed-in user's submitted jobs. */
export function useJobs(userId: string | null | undefined): JobsState {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [reloadToken, setReloadToken] = useState(0);

  const refresh = useCallback(() => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    let cancelled = false;
    const request = userId
      ? jobsApi.listJobs(userId)
      : Promise.resolve<Job[]>([]);

    void request.then((result) => {
      if (cancelled) return;
      setJobs(result);
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [userId, reloadToken]);

  // Reflect refetches in `loading` so callers can disable their refresh
  // controls and avoid firing overlapping requests.
  const refreshWithLoading = useCallback(() => {
    setLoading(true);
    refresh();
  }, [refresh]);

  return { jobs, loading, refresh: refreshWithLoading };
}
