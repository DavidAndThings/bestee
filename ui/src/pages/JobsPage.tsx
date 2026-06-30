import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@clerk/react";
import { Link } from "react-router-dom";
import { getSchema } from "../config/schemas";
import type { Job, JobStatus } from "../lib/types";
import { relativeTime } from "../lib/format";
import { JOBS_PAGE_SIZE, useJobs } from "../hooks/useJobs";
import { jobsApi } from "../services/jobsApi";
import refreshIcon from "../assets/icons/refresh.svg";

const STATUS_BADGE: Record<JobStatus, { label: string; className: string }> = {
  queued: { label: "Queued", className: "badge-ghost" },
  running: { label: "Running", className: "badge-info" },
  completed: { label: "Completed", className: "badge-success" },
  failed: { label: "Failed", className: "badge-error" },
};

/** One collapsible job row. The header summarises the job; expanding it reveals
 *  the job-log link, one link per result aspect (once completed), and Redo. */
function JobItem({
  job,
  onResubmit,
  resubmittingId,
}: {
  job: Job;
  onResubmit: (job: Job) => void;
  resubmittingId: string | null;
}) {
  const schema = getSchema(job.schemaId);
  const toolName = schema?.name ?? job.schemaId;
  const aspects = schema?.aspects ?? [];
  const badge = STATUS_BADGE[job.status];
  const resubmitting = resubmittingId === job.id;
  const busy = resubmittingId !== null;
  // The result tables only exist once the job completes; the result id is known
  // from submit time, so each aspect link is ready the moment the status flips.
  const showAspects = job.status === "completed" && Boolean(job.resultId);

  return (
    <details className="collapse-arrow bg-base-100 border-base-300 rounded-box collapse border">
      <summary className="collapse-title">
        <div className="flex items-center justify-between gap-3 pr-4">
          <div className="min-w-0">
            <p className="truncate font-semibold">{toolName}</p>
            <p className="text-base-content/60 mt-0.5 text-xs">
              {relativeTime(job.createdAt)} ·{" "}
              <span className="font-mono" title={job.id}>
                {job.id.slice(0, 8)}…
              </span>
            </p>
          </div>
          <span className={`badge shrink-0 ${badge.className}`}>
            {badge.label}
          </span>
        </div>
      </summary>
      <div className="collapse-content">
        <div className="flex flex-wrap items-center gap-2">
          <Link to={`/jobs/${job.id}`} className="btn btn-outline btn-xs">
            Job log
          </Link>
          {showAspects &&
            aspects.map((aspect) => (
              <Link
                key={aspect.name}
                to={`/results/${job.resultId}/${aspect.name}`}
                className="btn btn-outline btn-xs"
              >
                {aspect.label}
              </Link>
            ))}
          <button
            type="button"
            className="btn btn-ghost btn-xs"
            onClick={() => onResubmit(job)}
            disabled={busy}
          >
            {resubmitting ? (
              <span className="loading loading-spinner loading-xs" />
            ) : (
              "Redo"
            )}
          </button>
        </div>
      </div>
    </details>
  );
}

function JobsPage() {
  const { userId } = useAuth();
  const {
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
  } = useJobs(userId);
  const [resubmittingId, setResubmittingId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{
    kind: "success" | "error";
    message: string;
  } | null>(null);

  // Resubmit re-POSTs the job's stored schema + payload as a new run. The result
  // id is a deterministic hash of the request, so a redo overwrites the same
  // result slot rather than creating a divergent one.
  const handleResubmit = useCallback(
    (job: Job) => {
      if (!userId || resubmittingId) return;
      setResubmittingId(job.id);
      setFeedback(null);
      jobsApi
        .submitJob(userId, { schemaId: job.schemaId, payload: job.payload })
        .then(() => {
          setFeedback({ kind: "success", message: "Job resubmitted." });
          refresh();
        })
        .catch(() => {
          setFeedback({
            kind: "error",
            message: "Couldn't resubmit the job. Please try again.",
          });
        })
        .finally(() => setResubmittingId(null));
    },
    [userId, resubmittingId, refresh],
  );

  // Auto-dismiss the resubmit toast after a few seconds.
  useEffect(() => {
    if (!feedback) return;
    const timer = setTimeout(() => setFeedback(null), 4000);
    return () => clearTimeout(timer);
  }, [feedback]);

  return (
    <div className="p-4 sm:p-8">
      <div className="mx-auto max-w-4xl">
        <div className="mb-6 px-1">
          <div className="flex items-center justify-between gap-4">
            <h1 className="text-2xl font-semibold sm:text-3xl">Job Status</h1>
            <div className="tooltip tooltip-left" data-tip="Refresh">
              <button
                type="button"
                className="btn btn-ghost btn-circle btn-sm"
                onClick={refresh}
                disabled={loading}
                aria-label="Refresh"
              >
                <img src={refreshIcon} alt="" className="size-5" />
              </button>
            </div>
          </div>
        </div>

        {loading && jobs.length === 0 ? (
          <div className="flex justify-center p-12">
            <span className="loading loading-spinner loading-lg" />
          </div>
        ) : jobs.length === 0 ? (
          <div className="border-base-300 rounded-box border border-dashed p-10 text-center">
            <p className="text-base-content/60">No jobs yet.</p>
            <Link to="/" className="btn btn-primary btn-sm mt-4">
              Choose a chart
            </Link>
          </div>
        ) : (
          <div
            className={`space-y-3 transition-opacity ${
              loading ? "opacity-60" : ""
            }`}
          >
            {jobs.map((job) => (
              <JobItem
                key={job.id}
                job={job}
                onResubmit={handleResubmit}
                resubmittingId={resubmittingId}
              />
            ))}
          </div>
        )}

        {pageCount > 1 && (
          <div className="mt-6 flex flex-col items-center justify-between gap-3 sm:flex-row">
            <span className="text-base-content/60 text-sm">
              Showing {(page - 1) * JOBS_PAGE_SIZE + 1}–
              {(page - 1) * JOBS_PAGE_SIZE + jobs.length} of {total}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={prevPage}
                disabled={!hasPrev || loading}
              >
                Previous
              </button>
              <span className="text-base-content/70 px-1 text-sm whitespace-nowrap">
                Page {page} of {pageCount}
              </span>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={nextPage}
                disabled={!hasNext || loading}
              >
                Next
              </button>
            </div>
          </div>
        )}

        {feedback && (
          <div className="toast toast-end">
            <div
              className={`alert ${
                feedback.kind === "success" ? "alert-success" : "alert-error"
              }`}
            >
              <span>{feedback.message}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default JobsPage;
