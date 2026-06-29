import { useAuth } from "@clerk/react";
import { Link } from "react-router-dom";
import { getSchema } from "../config/schemas";
import type { Job, JobStatus } from "../lib/types";
import { relativeTime } from "../lib/format";
import { useJobs } from "../hooks/useJobs";
import refreshIcon from "../assets/icons/refresh.svg";
import cancelIcon from "../assets/icons/cancel.svg";
import infoIcon from "../assets/icons/info.svg";
import redoIcon from "../assets/icons/redo.svg";

const STATUS_BADGE: Record<JobStatus, { label: string; className: string }> = {
  queued: { label: "Queued", className: "badge-ghost" },
  running: { label: "Running", className: "badge-info" },
  completed: { label: "Completed", className: "badge-success" },
  failed: { label: "Failed", className: "badge-error" },
};

/** View table / Detail / Redo / Cancel action buttons.  Shared by the desktop
 *  table row and the mobile card so we don't drift the per-status logic. */
function JobActions({ job }: { job: Job }) {
  const aspect = getSchema(job.schemaId)?.aspect;
  // The result table only exists once the job completes; the result id is known
  // from submit time, so the link is ready the moment the status flips.
  const tableHref =
    job.status === "completed" && job.resultId && aspect
      ? `/results/${job.resultId}/${aspect}`
      : null;
  return (
    <div className="flex flex-nowrap items-center justify-end gap-2">
      {tableHref && (
        <Link to={tableHref} className="btn btn-outline btn-xs">
          View table
        </Link>
      )}
      {job.status !== "queued" && (
        <div className="tooltip tooltip-left" data-tip="Details">
          <button
            type="button"
            className="btn btn-ghost btn-circle btn-sm"
            aria-label="Details"
          >
            <img src={infoIcon} alt="" className="size-6" />
          </button>
        </div>
      )}
      <div className="tooltip tooltip-left" data-tip="Redo">
        <button
          type="button"
          className="btn btn-ghost btn-circle btn-sm"
          aria-label="Redo"
        >
          <img src={redoIcon} alt="" className="size-6" />
        </button>
      </div>
      {(job.status === "queued" || job.status === "running") && (
        <div className="tooltip tooltip-left" data-tip="Cancel">
          <button
            type="button"
            className="btn btn-ghost btn-circle btn-sm"
            aria-label="Cancel"
          >
            <img src={cancelIcon} alt="" className="size-6" />
          </button>
        </div>
      )}
    </div>
  );
}

function JobsPage() {
  const { userId } = useAuth();
  const { jobs, loading, refresh } = useJobs(userId);

  return (
    <div className="p-4 sm:p-8">
      <div className="mx-auto max-w-6xl">
        <div className="mb-6 px-4">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold sm:text-3xl">Job Status</h1>
            </div>
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
          <>
            {/* Mobile: card-per-row.  The desktop table doesn't fit
             *  under ~700px once the UUID + actions are in. */}
            <ul className="space-y-3 md:hidden">
              {jobs.map((job) => {
                const status = STATUS_BADGE[job.status];
                const toolName = getSchema(job.schemaId)?.name ?? job.schemaId;
                return (
                  <li
                    key={job.id}
                    className="bg-base-100 border-base-300 rounded-box border p-4 shadow-sm"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="font-semibold">{toolName}</p>
                        <p
                          className="text-base-content/60 mt-1 font-mono text-xs"
                          title={job.id}
                        >
                          {job.id.slice(0, 8)}…
                        </p>
                        <p className="text-base-content/60 mt-1 text-sm">
                          {relativeTime(job.createdAt)}
                        </p>
                      </div>
                      <span className={`badge shrink-0 ${status.className}`}>
                        {status.label}
                      </span>
                    </div>
                    <div className="mt-3 flex justify-end">
                      <JobActions job={job} />
                    </div>
                  </li>
                );
              })}
            </ul>

            {/* Desktop: tabular view. */}
            <div className="hidden overflow-x-auto md:block">
              <table className="table">
                <thead>
                  <tr>
                    <th>Job ID</th>
                    <th>Tool</th>
                    <th>Status</th>
                    <th>Submitted</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => {
                    const status = STATUS_BADGE[job.status];
                    const toolName =
                      getSchema(job.schemaId)?.name ?? job.schemaId;
                    return (
                      <tr key={job.id}>
                        <td className="font-mono text-xs">{job.id}</td>
                        <td>{toolName}</td>
                        <td>
                          <span className={`badge ${status.className}`}>
                            {status.label}
                          </span>
                        </td>
                        <td className="text-base-content/60 whitespace-nowrap">
                          {relativeTime(job.createdAt)}
                        </td>
                        <td className="text-right">
                          <JobActions job={job} />
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default JobsPage;
