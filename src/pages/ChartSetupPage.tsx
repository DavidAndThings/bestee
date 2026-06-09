import { type FormEvent, useMemo, useState } from "react";
import { useAuth } from "@clerk/react";
import { Link, Navigate, useParams } from "react-router-dom";
import { getSchema, type FieldDef } from "../config/schemas";
import { humanizeKey } from "../lib/format";
import { jobsApi } from "../services/jobsApi";

type FormValues = Record<string, string>;
type FormErrors = Record<string, string>;

type SubmitState =
  | { status: "idle" }
  | { status: "submitting" }
  | { status: "success"; jobId: string }
  | { status: "error"; message: string };

function initialValues(parameters: Record<string, FieldDef>): FormValues {
  return Object.fromEntries(Object.keys(parameters).map((key) => [key, ""]));
}

function parseArray(raw: string): string[] {
  return raw
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function coerceValue(
  key: string,
  def: FieldDef,
  raw: string,
): { value?: unknown; error?: string } {
  const trimmed = raw.trim();

  if (def.type === "array") {
    const items = parseArray(raw);
    if (items.length === 0) {
      return { error: `${humanizeKey(key)} needs at least one value.` };
    }
    return { value: items };
  }

  if (!trimmed) {
    return { error: `${humanizeKey(key)} is required.` };
  }

  if (def.type === "integer") {
    const number = Number(trimmed);
    if (!Number.isFinite(number) || !Number.isInteger(number)) {
      return { error: `${humanizeKey(key)} must be a whole number.` };
    }
    return { value: number };
  }

  if (def.choices && !def.choices.includes(trimmed)) {
    return { error: `Choose a valid ${humanizeKey(key)} option.` };
  }

  return { value: trimmed };
}

function buildPayload(
  parameters: Record<string, FieldDef>,
  values: FormValues,
): { payload: Record<string, unknown>; errors: FormErrors } {
  const payload: Record<string, unknown> = {};
  const errors: FormErrors = {};

  for (const [key, def] of Object.entries(parameters)) {
    const result = coerceValue(key, def, values[key] ?? "");
    if (result.error) {
      errors[key] = result.error;
    } else {
      payload[key] = result.value;
    }
  }

  return { payload, errors };
}

function ChartSetupPage() {
  const { schemaId } = useParams();
  const schema = getSchema(schemaId);
  const { userId } = useAuth();
  const [values, setValues] = useState<FormValues>(() =>
    schema ? initialValues(schema.parameters) : {},
  );
  const [errors, setErrors] = useState<FormErrors>({});
  const [submitState, setSubmitState] = useState<SubmitState>({
    status: "idle",
  });

  const entries = useMemo(
    () => (schema ? Object.entries(schema.parameters) : []),
    [schema],
  );

  if (!schema) {
    return <Navigate to="/" replace />;
  }

  const updateValue = (key: string, value: string) => {
    setValues((prev) => ({ ...prev, [key]: value }));
    setErrors((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    if (submitState.status !== "idle") {
      setSubmitState({ status: "idle" });
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!userId || submitState.status === "submitting") return;

    const result = buildPayload(schema.parameters, values);
    setErrors(result.errors);
    if (Object.keys(result.errors).length > 0) return;

    setSubmitState({ status: "submitting" });
    try {
      const response = await jobsApi.submitJob(userId, {
        schemaId: schema.id,
        payload: result.payload,
      });
      setSubmitState({ status: "success", jobId: response.requestId });
    } catch {
      setSubmitState({
        status: "error",
        message: "Something went wrong creating the job. Please try again.",
      });
    }
  };

  return (
    <div className="p-8">
      <div className="mx-auto max-w-3xl">
        <div className="mb-6">
          <Link to="/" className="link link-hover text-base-content/60 text-sm">
            ← Back to Home
          </Link>
          <div className="mt-4 flex flex-wrap items-start justify-between gap-4">
            <div>
              <h1 className="text-3xl font-semibold">{schema.name}</h1>
              <p className="text-base-content/70 mt-2 max-w-2xl">
                {schema.description}
              </p>
            </div>
            {schema.badge && (
              <span className="badge badge-primary badge-outline">
                {schema.badge}
              </span>
            )}
          </div>
        </div>

        <form onSubmit={handleSubmit} className="card bg-base-100 shadow-md">
          <div className="card-body gap-7 sm:gap-8">
            {entries.map(([key, def]) => {
              const id = `field-${key}`;
              const error = errors[key];
              return (
                <label
                  key={key}
                  className="form-control w-full gap-4"
                  htmlFor={id}
                >
                  <div className="label px-0 py-0">
                    <span className="label-text text-sm font-semibold tracking-wide">
                      {humanizeKey(key)}
                    </span>
                  </div>

                  {def.choices ? (
                    <select
                      id={id}
                      className={`select select-bordered w-full ${
                        error ? "select-error" : ""
                      }`}
                      value={values[key] ?? ""}
                      onChange={(event) => updateValue(key, event.target.value)}
                    >
                      <option value="" disabled>
                        Select an option
                      </option>
                      {def.choices.map((choice) => (
                        <option key={choice} value={choice}>
                          {choice}
                        </option>
                      ))}
                    </select>
                  ) : def.type === "array" ? (
                    <textarea
                      id={id}
                      className={`textarea textarea-bordered min-h-28 w-full ${
                        error ? "textarea-error" : ""
                      }`}
                      value={values[key] ?? ""}
                      placeholder="AAPL, MSFT, GOOG"
                      onChange={(event) => updateValue(key, event.target.value)}
                    />
                  ) : (
                    <input
                      id={id}
                      type={def.type === "integer" ? "number" : "text"}
                      step={def.type === "integer" ? 1 : undefined}
                      className={`input input-bordered w-full ${
                        error ? "input-error" : ""
                      }`}
                      value={values[key] ?? ""}
                      onChange={(event) => updateValue(key, event.target.value)}
                    />
                  )}

                  <div className="label flex-col items-start gap-1 px-0 py-0 sm:flex-row sm:items-center">
                    <span className="label-text-alt text-base-content/60 leading-relaxed">
                      {def.description}
                      {def.type === "array" &&
                        " Separate values with commas or new lines."}
                    </span>
                    {error && (
                      <span className="label-text-alt text-error">{error}</span>
                    )}
                  </div>
                </label>
              );
            })}

            {submitState.status === "success" && (
              <div className="alert alert-success">
                <span>
                  Job created. ID:{" "}
                  <span className="font-mono">{submitState.jobId}</span>
                </span>
                <Link to="/jobs" className="btn btn-sm">
                  View jobs
                </Link>
              </div>
            )}

            {submitState.status === "error" && (
              <div className="alert alert-error">
                <span>{submitState.message}</span>
              </div>
            )}

            <div className="card-actions justify-end">
              <button
                type="submit"
                className="btn btn-primary"
                disabled={submitState.status === "submitting"}
              >
                {submitState.status === "submitting" && (
                  <span className="loading loading-spinner loading-sm" />
                )}
                Create job
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

export default ChartSetupPage;
