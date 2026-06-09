import { useAuth } from "@clerk/react";
import { Link } from "react-router-dom";
import { getSchema } from "../config/schemas";
import type { JobStatus } from "../lib/types";
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

function JobsPage() {
  const { userId } = useAuth();
  const { jobs, loading, refresh } = useJobs(userId);

  return (
    <div className="p-8">
      <div className="mx-auto max-w-4xl">
        <div className="mb-6 px-4">
          <Link to="/" className="link link-hover text-base-content/60 text-sm">
            ← Back to Home
          </Link>
          <div className="mt-4 flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold">Job Status</h1>
            </div>
            <div className="tooltip tooltip-left" data-tip="refresh">
              <button
                type="button"
                className="btn btn-ghost btn-circle btn-sm"
                onClick={refresh}
                disabled={loading}
                aria-label="refresh"
              >
                <img src={refreshIcon} alt="" className="size-5" />
              </button>
            </div>
          </div>
        </div>

        {loading ? (
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
          <div className="overflow-x-auto">
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
                        <div className="flex items-center justify-end gap-2">
                          {job.status !== "queued" && (
                            <div
                              className="tooltip tooltip-left"
                              data-tip="Details"
                            >
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
                          {(job.status === "queued" ||
                            job.status === "running") && (
                            <div
                              className="tooltip tooltip-left"
                              data-tip="Cancel"
                            >
                              <button
                                type="button"
                                className="btn btn-ghost btn-circle btn-sm"
                                aria-label="Cancel"
                              >
                                <img
                                  src={cancelIcon}
                                  alt=""
                                  className="size-6"
                                />
                              </button>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default JobsPage;
