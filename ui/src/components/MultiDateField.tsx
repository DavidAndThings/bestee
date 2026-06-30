import DateField from "./DateField";

type Props = {
  id?: string;
  /** The selected dates as ISO `yyyy-mm-dd` strings (empty rows allowed). */
  values: string[];
  onChange: (values: string[]) => void;
  invalid?: boolean;
  autoFocus?: boolean;
};

/**
 * A repeatable list of {@link DateField}s for collecting one or more dates
 * (e.g. several out-of-sample prediction dates). Each row has its own calendar
 * and a remove button; "Add date" appends an empty row. Reads and writes a
 * `string[]` of ISO dates -- blank rows are kept in state and filtered by the
 * form on submit.
 */
export default function MultiDateField({
  id,
  values,
  onChange,
  invalid,
  autoFocus,
}: Props) {
  const setAt = (index: number, value: string) => {
    const next = values.slice();
    next[index] = value;
    onChange(next);
  };
  const removeAt = (index: number) =>
    onChange(values.filter((_, i) => i !== index));
  const add = () => onChange([...values, ""]);

  return (
    <div className="w-full space-y-2">
      {values.map((value, index) => (
        <div key={index} className="flex items-center gap-2">
          <div className="min-w-0 flex-1">
            <DateField
              // The label points at the first picker (or the add button when
              // empty), so it always targets a real, labelable control.
              id={index === 0 ? id : undefined}
              value={value}
              onChange={(next) => setAt(index, next)}
              invalid={invalid}
              autoFocus={autoFocus && index === 0}
            />
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm btn-circle text-base-content/50 hover:text-error shrink-0"
            aria-label={`Remove date ${index + 1}`}
            onClick={() => removeAt(index)}
          >
            ✕
          </button>
        </div>
      ))}
      <button
        type="button"
        id={values.length === 0 ? id : undefined}
        className="btn btn-ghost btn-sm gap-1"
        onClick={add}
      >
        <span aria-hidden="true">+</span> Add date
      </button>
    </div>
  );
}
