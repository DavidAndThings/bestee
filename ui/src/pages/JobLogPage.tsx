import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getSchema } from "../config/schemas";
import { humanizeKey } from "../lib/format";
import type { JobDetail, JobStatus } from "../lib/types";
import { jobsApi } from "../services/jobsApi";

const STATUS_BADGE: Record<JobStatus, { label: string; className: string }> = {
  queued: { label: "Queued", className: "badge-ghost" },
  running: { label: "Running", className: "badge-info" },
  completed: { label: "Completed", className: "badge-success" },
  failed: { label: "Failed", className: "badge-error" },
};

type LoadState =
  | { status: "loading" }
  | { status: "ready"; detail: JobDetail }
  | { status: "error"; message: string };

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? iso : new Date(ms).toLocaleString();
}

/** Render an input value: primitive arrays as a list, objects as JSON. */
function formatInput(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (Array.isArray(value)) {
    return value.every((item) => typeof item !== "object" || item === null)
      ? value.join(", ")
      : JSON.stringify(value);
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function Field({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-col">
      <dt className="text-base-content/50 text-xs tracking-wide uppercase">
        {label}
      </dt>
      <dd className={`break-all ${mono ? "font-mono text-sm" : "text-sm"}`}>
        {value}
      </dd>
    </div>
  );
}

function JobLog({ detail }: { detail: JobDetail }) {
  const tool = getSchema(detail.schemaId)?.name ?? detail.schemaId ?? "Analysis";
  const badge = STATUS_BADGE[detail.status];
  const inputs = Object.entries(detail.payload ?? {});
  return (
    <div className="space-y-6">
      <div className="card bg-base-100 border-base-300 border shadow-sm">
        <div className="card-body gap-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="card-title">{tool}</h2>
            <span className={`badge ${badge.className}`}>{badge.label}</span>
          </div>
          <dl className="grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
            <Field label="State" value={detail.state} mono />
            <Field
              label="Elapsed"
              value={
                detail.elapsedSeconds != null
                  ? `${detail.elapsedSeconds.toFixed(1)}s`
                  : "—"
              }
            />
            <Field label="Submitted" value={formatTime(detail.createdAt)} />
            <Field label="Started" value={formatTime(detail.startedAt)} />
            <Field label="Finished" value={formatTime(detail.finishedAt)} />
            <Field label="Result id" value={detail.resultId ?? "—"} mono />
          </dl>
        </div>
      </div>

      {inputs.length > 0 && (
        <div className="card bg-base-100 border-base-300 border shadow-sm">
          <div className="card-body gap-3">
            <h3 className="font-semibold">Inputs</h3>
            <dl className="grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
              {inputs.map(([key, value]) => (
                <Field
                  key={key}
                  label={humanizeKey(key)}
                  value={formatInput(value)}
                />
              ))}
            </dl>
          </div>
        </div>
      )}

      {(detail.error || detail.traceback) && (
        <div className="card bg-base-100 border-error/40 border shadow-sm">
          <div className="card-body gap-3">
            <h3 className="text-error font-semibold">Error</h3>
            {detail.error && <p className="text-sm">{detail.error}</p>}
            {detail.traceback && (
              <pre className="bg-base-200 rounded-box max-h-96 overflow-auto p-4 text-xs whitespace-pre-wrap">
                {detail.traceback}
              </pre>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/** The log / detail view for one job (`/jobs/:taskId`). */
function JobLogPage() {
  const { taskId } = useParams();
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    jobsApi
      .getJob(taskId)
      .then((detail) => {
        if (!cancelled) setState({ status: "ready", detail });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const message =
          error instanceof Error ? error.message : "Failed to load.";
        setState({ status: "error", message });
      });
    return () => {
      cancelled = true;
    };
  }, [taskId]);

  return (
    <div className="p-4 sm:p-8">
      <div className="mx-auto max-w-4xl">
        <div className="mb-6 flex items-start justify-between gap-4 px-1">
          <div className="min-w-0">
            <h1 className="text-2xl font-semibold sm:text-3xl">Job log</h1>
            <p
              className="text-base-content/60 mt-1 font-mono text-xs break-all"
              title={taskId}
            >
              {taskId}
            </p>
          </div>
          <Link to="/jobs" className="btn btn-ghost btn-sm shrink-0">
            Back to jobs
          </Link>
        </div>

        {state.status === "loading" && (
          <div className="flex justify-center p-12">
            <span className="loading loading-spinner loading-lg" />
          </div>
        )}

        {state.status === "error" && (
          <div className="border-base-300 rounded-box border border-dashed p-10 text-center">
            <p className="text-base-content/70">Couldn't load this job.</p>
            <p className="text-base-content/50 mt-1 text-sm">{state.message}</p>
            <Link to="/jobs" className="btn btn-primary btn-sm mt-4">
              Back to jobs
            </Link>
          </div>
        )}

        {state.status === "ready" && <JobLog detail={state.detail} />}
      </div>
    </div>
  );
}

export default JobLogPage;
