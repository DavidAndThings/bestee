import { useCallback, useEffect, useState } from "react";
import type { Job } from "../lib/types";
import { jobsApi } from "../services/jobsApi";

/** Jobs shown per page in the Job Status list. */
export const JOBS_PAGE_SIZE = 10;

export type JobsState = {
  jobs: Job[];
  loading: boolean;
  refresh: () => void;
  /** Total jobs across all pages. */
  total: number;
  /** 1-based current page. */
  page: number;
  /** Total number of pages (>= 1). */
  pageCount: number;
  hasPrev: boolean;
  hasNext: boolean;
  nextPage: () => void;
  prevPage: () => void;
};

/** Loads one page of the signed-in user's submitted jobs (newest first). */
export function useJobs(userId: string | null | undefined): JobsState {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [reloadToken, setReloadToken] = useState(0);

  // `offset` is session-local: it starts at the first page and only moves via
  // next/prev. A sign-in/out routes away and remounts this page (resetting it),
  // and the initial `undefined -> userId` transition already loads page one.
  useEffect(() => {
    let cancelled = false;
    const request = userId
      ? jobsApi.listJobs(userId, offset, JOBS_PAGE_SIZE)
      : Promise.resolve({
          jobs: [],
          total: 0,
          offset: 0,
          limit: JOBS_PAGE_SIZE,
        });

    void request.then((result) => {
      if (cancelled) return;
      setJobs(result.jobs);
      setTotal(result.total);
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [userId, offset, reloadToken]);

  // Reflect refetches in `loading` so callers can disable their refresh and
  // paging controls and avoid firing overlapping requests.
  const refresh = useCallback(() => {
    setLoading(true);
    setReloadToken((token) => token + 1);
  }, []);

  const pageCount = Math.max(1, Math.ceil(total / JOBS_PAGE_SIZE));
  const page = Math.floor(offset / JOBS_PAGE_SIZE) + 1;
  const hasPrev = offset > 0;
  const hasNext = offset + JOBS_PAGE_SIZE < total;

  const nextPage = useCallback(() => {
    setLoading(true);
    setOffset((prev) => prev + JOBS_PAGE_SIZE);
  }, []);
  const prevPage = useCallback(() => {
    setLoading(true);
    setOffset((prev) => Math.max(0, prev - JOBS_PAGE_SIZE));
  }, []);

  return {
    jobs,
    loading,
    refresh,
    total,
    page,
    pageCount,
    hasPrev,
    hasNext,
    nextPage,
    prevPage,
  };
}
