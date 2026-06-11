import { type FormEvent, useMemo, useState } from "react";
import { useAuth } from "@clerk/react";
import { Link, Navigate, useParams } from "react-router-dom";
import { getSchema, type FieldDef, type FieldType } from "../config/schemas";
import { humanizeKey } from "../lib/format";
import { jobsApi } from "../services/jobsApi";
import DateField from "../components/DateField";

type FormValues = Record<string, string>;
type FormErrors = Record<string, string>;

type SubmitState =
  | { status: "idle" }
  | { status: "submitting" }
  | { status: "success"; jobId: string }
  | { status: "error"; message: string };

// These fields hold tool parameters, never credentials — keep password
// managers and browser autofill from offering to fill them.
const noAutofill = {
  autoComplete: "off",
  "data-1p-ignore": "true", // 1Password
  "data-lpignore": "true", // LastPass
  "data-bwignore": "true", // Bitwarden
  "data-form-type": "other", // Dashlane
};

function initialValues(parameters: Record<string, FieldDef>): FormValues {
  return Object.fromEntries(Object.keys(parameters).map((key) => [key, ""]));
}

function parseArray(raw: string): string[] {
  return raw
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

/** The HTML input `type` for a free-entry text/number field. `date` fields are
 * rendered separately with a calendar (see DateField). */
function inputType(type: FieldType): string {
  return type === "integer" ? "number" : "text";
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

  if (def.type === "date") {
    // A native date input yields an ISO yyyy-mm-dd string.
    if (
      !/^\d{4}-\d{2}-\d{2}$/.test(trimmed) ||
      Number.isNaN(Date.parse(trimmed))
    ) {
      return { error: `${humanizeKey(key)} must be a valid date.` };
    }
    return { value: trimmed };
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
    // Only touch error state when there's actually an error to clear,
    // avoiding a second re-render on every keystroke.
    setErrors((prev) => {
      if (!(key in prev)) return prev;
      const next = { ...prev };
      delete next[key];
      return next;
    });
    // Clear a submit error when the user starts editing in response
    // to it; leave a success alert alone — it's informational, the
    // next submit will naturally overwrite it.
    if (submitState.status === "error") {
      setSubmitState({ status: "idle" });
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!userId || submitState.status === "submitting") return;

    const result = buildPayload(schema.parameters, values);
    const fieldErrors = { ...result.errors };
    // Cross-field checks only run once every field is individually valid.
    if (Object.keys(fieldErrors).length === 0 && schema.validate) {
      Object.assign(fieldErrors, schema.validate(result.payload));
    }
    setErrors(fieldErrors);
    if (Object.keys(fieldErrors).length > 0) return;

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
    <div className="p-4 sm:p-8">
      <div className="mx-auto max-w-4xl">
        <form onSubmit={handleSubmit} className="card bg-base-100 shadow-md">
          <div className="card-body gap-7 sm:gap-8">
            <div className="border-base-300 border-b pb-6">
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="min-w-0">
                  <h1 className="card-title wrap-break-word text-2xl sm:text-3xl">
                    {schema.name}
                  </h1>
                  <p className="text-base-content/70 wrap-break-word mt-2 max-w-2xl">
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
            {entries.map(([key, def], index) => {
              const id = `field-${key}`;
              const error = errors[key];
              // First field receives focus on mount so keyboard users
              // can start typing immediately.
              const autoFocus = index === 0;
              return (
                <div key={key} className="form-control w-full">
                  <label
                    htmlFor={id}
                    className="label-text mb-5 block text-base font-semibold tracking-wide"
                  >
                    {humanizeKey(key)}
                  </label>

                  {def.choices ? (
                    <select
                      id={id}
                      {...noAutofill}
                      autoFocus={autoFocus}
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
                      {...noAutofill}
                      autoFocus={autoFocus}
                      className={`textarea textarea-bordered min-h-28 w-full ${
                        error ? "textarea-error" : ""
                      }`}
                      value={values[key] ?? ""}
                      placeholder="AAPL, MSFT, GOOG"
                      onChange={(event) => updateValue(key, event.target.value)}
                    />
                  ) : def.type === "date" ? (
                    <DateField
                      id={id}
                      value={values[key] ?? ""}
                      onChange={(next) => updateValue(key, next)}
                      invalid={!!error}
                      autoFocus={autoFocus}
                    />
                  ) : (
                    <input
                      id={id}
                      {...noAutofill}
                      autoFocus={autoFocus}
                      type={inputType(def.type)}
                      step={def.type === "integer" ? 1 : undefined}
                      className={`input input-bordered w-full ${
                        error ? "input-error" : ""
                      }`}
                      value={values[key] ?? ""}
                      onChange={(event) => updateValue(key, event.target.value)}
                    />
                  )}

                  <div className="mt-5 flex flex-col items-start gap-2 sm:flex-row sm:items-center">
                    <span className="text-base-content/60 text-sm leading-relaxed">
                      {def.description}
                      {def.type === "array" &&
                        " Separate values with commas or new lines."}
                    </span>
                    {error && (
                      <span className="text-error text-sm">{error}</span>
                    )}
                  </div>
                </div>
              );
            })}

            {submitState.status === "success" && (
              <div className="alert alert-success flex-col items-start sm:flex-row sm:items-center">
                <span>
                  Job created. ID:{" "}
                  <span className="break-all font-mono">
                    {submitState.jobId}
                  </span>
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
