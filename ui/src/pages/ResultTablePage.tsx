import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { jobsApi } from "../services/jobsApi";
import type { ResultTable } from "../lib/types";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; table: ResultTable }
  | { status: "error"; message: string };

/** Render a table cell value: integers as-is, other numbers fixed, nulls as a dash. */
function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  return String(value);
}

/** View one analysis-specific result aspect (`/results/:resultId/:aspect`) as a table. */
function ResultTablePage() {
  const { resultId, aspect } = useParams();
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    if (!resultId || !aspect) return;
    let cancelled = false;
    jobsApi
      .getResultAspect(resultId, aspect)
      .then((table) => {
        if (!cancelled) setState({ status: "ready", table });
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
  }, [resultId, aspect]);

  const title = aspect ? aspect.replace(/_/g, " ") : "Result";

  return (
    <div className="p-4 sm:p-8">
      <div className="mx-auto max-w-4xl">
        <div className="mb-6 flex items-start justify-between gap-4 px-1">
          <div className="min-w-0">
            <h1 className="text-2xl font-semibold capitalize sm:text-3xl">
              {title}
            </h1>
            <p
              className="text-base-content/60 mt-1 font-mono text-xs break-all"
              title={resultId}
            >
              {resultId}
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
            <p className="text-base-content/70">
              This result isn't available yet — the job may still be running.
            </p>
            <p className="text-base-content/50 mt-1 text-sm">{state.message}</p>
            <Link to="/jobs" className="btn btn-primary btn-sm mt-4">
              Back to jobs
            </Link>
          </div>
        )}

        {state.status === "ready" &&
          (state.table.rows.length === 0 ? (
            <div className="border-base-300 rounded-box border border-dashed p-10 text-center">
              <p className="text-base-content/60">This table is empty.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="table-zebra table">
                <thead>
                  <tr>
                    {state.table.columns.map((column) => (
                      <th key={column}>{column}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {state.table.rows.map((row, index) => (
                    <tr key={index}>
                      {state.table.columns.map((column) => (
                        <td key={column} className="font-mono text-xs">
                          {formatCell(row[column])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
      </div>
    </div>
  );
}

export default ResultTablePage;
