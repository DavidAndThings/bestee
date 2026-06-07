import type { Schema } from "../../config/schemas";
import type { Question } from "../../lib/types";
import { buildPayload } from "../../lib/buildPayload";
import { formatValue, humanizeKey } from "../../lib/format";

export type ConfirmationCardProps = {
  schema: Schema;
  questions: Question[];
  values: Record<string, unknown>;
  submitting?: boolean;
  onConfirm: () => void;
  onEditField: (fieldKey: string) => void;
};

function ConfirmationCard({
  schema,
  questions,
  values,
  submitting,
  onConfirm,
  onEditField,
}: ConfirmationCardProps) {
  const json = JSON.stringify(buildPayload(schema, values), null, 2);

  return (
    <div className="card bg-base-200 border-base-300 border">
      <div className="card-body gap-4 p-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="font-semibold">Ready to run: {schema.name}</h3>
          <span className="badge badge-warning badge-sm">Review</span>
        </div>

        <div className="overflow-x-auto">
          <table className="table table-sm">
            <tbody>
              {questions.map((question) => (
                <tr key={question.fieldKey}>
                  <td className="text-base-content/60 w-2/5 align-top">
                    {humanizeKey(question.fieldKey)}
                  </td>
                  <td className="font-medium">
                    {formatValue(values[question.fieldKey])}
                  </td>
                  <td className="text-right">
                    <button
                      type="button"
                      className="btn btn-ghost btn-xs"
                      disabled={submitting}
                      onClick={() => onEditField(question.fieldKey)}
                    >
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <details className="collapse-arrow bg-base-100 border-base-300 collapse border">
          <summary className="collapse-title text-sm font-medium">
            View JSON payload
          </summary>
          <div className="collapse-content">
            <pre className="overflow-x-auto text-xs">
              <code>{json}</code>
            </pre>
          </div>
        </details>

        <button
          type="button"
          className="btn btn-primary self-start"
          disabled={submitting}
          onClick={onConfirm}
        >
          {submitting && (
            <span className="loading loading-spinner loading-sm" />
          )}
          Confirm &amp; run
        </button>
      </div>
    </div>
  );
}

export default ConfirmationCard;
