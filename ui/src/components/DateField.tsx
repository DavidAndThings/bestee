import { useEffect, useRef, useState } from "react";
import { DayPicker } from "react-day-picker";
import "react-day-picker/style.css";

type Props = {
  id?: string;
  /** Current value as an ISO `yyyy-mm-dd` string, or "" when empty. */
  value: string;
  onChange: (value: string) => void;
  invalid?: boolean;
  autoFocus?: boolean;
};

/** Parse an ISO `yyyy-mm-dd` string into a local Date (or undefined). */
function fromISO(value: string): Date | undefined {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return undefined;
  const [year, month, day] = value.split("-").map(Number);
  const date = new Date(year, month - 1, day);
  return Number.isNaN(date.getTime()) ? undefined : date;
}

/** Format a local Date as an ISO `yyyy-mm-dd` string. */
function toISO(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/**
 * A date field backed by a pop-over calendar (react-day-picker). Month/year
 * dropdowns keep any date reachable. Reads and writes ISO `yyyy-mm-dd` strings
 * to fit the form's string-based state.
 */
export default function DateField({
  id,
  value,
  onChange,
  invalid,
  autoFocus,
}: Props) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const selected = fromISO(value);

  // Close the calendar when clicking anywhere outside this field.
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  return (
    <div ref={containerRef} className="relative w-full">
      <button
        type="button"
        id={id}
        autoFocus={autoFocus}
        aria-haspopup="dialog"
        aria-expanded={open}
        className={`input input-bordered flex w-full cursor-pointer items-center justify-start text-left font-normal ${
          invalid ? "input-error" : ""
        }`}
        onClick={() => setOpen((prev) => !prev)}
      >
        {selected ? (
          selected.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric",
          })
        ) : (
          <span className="text-base-content/40">Pick a date</span>
        )}
      </button>
      {open && (
        <div className="border-base-300 bg-base-100 rounded-box absolute top-full left-0 z-30 mt-1 border p-2 shadow-lg">
          <DayPicker
            mode="single"
            selected={selected}
            defaultMonth={selected}
            captionLayout="dropdown"
            startMonth={new Date(1970, 0)}
            endMonth={new Date(2099, 11)}
            onSelect={(date) => {
              onChange(date ? toISO(date) : "");
              setOpen(false);
            }}
          />
        </div>
      )}
    </div>
  );
}
