import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { jobsApi } from "../services/jobsApi";
import type { SearchResult } from "../lib/types";

type Props = {
  id?: string;
  /** Selected terms: ticker symbols or SIC industry titles. */
  values: string[];
  onChange: (values: string[]) => void;
  invalid?: boolean;
  autoFocus?: boolean;
  /** Cap selection at a single term (e.g. an RRG benchmark). */
  single?: boolean;
  placeholder?: string;
};

const SEARCH_DEBOUNCE_MS = 200;

// Tool parameters, never credentials — keep password managers and browser
// autofill from offering to fill the search box.
const noAutofill = {
  autoComplete: "off",
  "data-1p-ignore": "true",
  "data-lpignore": "true",
  "data-bwignore": "true",
  "data-form-type": "other",
};

/**
 * A multi-select autocomplete over the `/search` catalog: securities (matched
 * by ticker symbol or company/ETF name, shown as "Name (SYMBOL)") and SIC
 * industry titles. Users type to search, pick one or more matches that show as
 * removable chips, and the backend later resolves each stored value (a symbol
 * or industry title) to ticker symbols. Set `single` for one-value fields.
 * Reads and writes a `string[]` of the selected values.
 */
export default function TickerSearchField({
  id,
  values,
  onChange,
  invalid,
  autoFocus,
  single = false,
  placeholder = "Search by symbol, name, or industry…",
}: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listboxId = useId();

  // Don't offer matches whose value is already selected.
  const visible = useMemo(
    () =>
      results.filter(
        (result) =>
          !values.some((v) => v.toLowerCase() === result.value.toLowerCase()),
      ),
    [results, values],
  );

  // Debounced search whenever the query changes. Every state update happens
  // inside the timeout/promise (never synchronously in the effect body) to
  // avoid cascading renders; `loading` is flipped on in the input handler.
  useEffect(() => {
    const trimmed = query.trim();
    let cancelled = false;
    const handle = setTimeout(() => {
      if (cancelled) return;
      if (!trimmed) {
        setResults([]);
        setLoading(false);
        return;
      }
      void jobsApi
        .search(trimmed)
        .then((res) => {
          if (cancelled) return;
          setResults(res);
          setFailed(false);
          setHighlight(0);
          setLoading(false);
        })
        .catch(() => {
          if (cancelled) return;
          setResults([]);
          setFailed(true);
          setLoading(false);
        });
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [query]);

  // Close the dropdown on any outside pointer interaction.
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

  const addTerm = (term: string) => {
    const exists = values.some((v) => v.toLowerCase() === term.toLowerCase());
    if (single) {
      onChange([term]);
    } else if (!exists) {
      onChange([...values, term]);
    }
    setQuery("");
    setResults([]);
    setOpen(false);
    inputRef.current?.focus();
  };

  const removeTerm = (term: string) => {
    onChange(values.filter((v) => v !== term));
    inputRef.current?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setOpen(true);
      setHighlight((h) => Math.min(h + 1, visible.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlight((h) => Math.max(h - 1, 0));
    } else if (event.key === "Enter") {
      // While searching, Enter commits a choice instead of submitting the form.
      if (query !== "" || open) {
        event.preventDefault();
        if (visible.length > 0) {
          const index =
            highlight >= 0 && highlight < visible.length ? highlight : 0;
          addTerm(visible[index].value);
        }
      }
    } else if (event.key === "Escape") {
      setOpen(false);
    } else if (event.key === "Backspace" && query === "" && values.length > 0) {
      removeTerm(values[values.length - 1]);
    }
  };

  const showDropdown = open && query.trim() !== "";
  const activeOptionId =
    showDropdown && !loading && visible.length > 0
      ? `${listboxId}-opt-${Math.min(highlight, visible.length - 1)}`
      : undefined;

  return (
    <div ref={containerRef} className="relative w-full">
      <div
        className={`bg-base-100 flex w-full flex-wrap items-center gap-2 rounded-lg border px-3 py-2 ${
          invalid
            ? "border-error"
            : "border-base-300 focus-within:border-primary"
        }`}
        onClick={() => inputRef.current?.focus()}
      >
        {values.map((term) => (
          <span key={term} className="badge badge-primary badge-outline gap-1">
            <span className="max-w-[16rem] truncate">{term}</span>
            <button
              type="button"
              className="hover:text-error cursor-pointer leading-none"
              aria-label={`Remove ${term}`}
              onClick={(event) => {
                event.stopPropagation();
                removeTerm(term);
              }}
            >
              ✕
            </button>
          </span>
        ))}
        <input
          id={id}
          ref={inputRef}
          {...noAutofill}
          autoFocus={autoFocus}
          type="text"
          role="combobox"
          aria-expanded={showDropdown}
          aria-controls={listboxId}
          aria-autocomplete="list"
          aria-activedescendant={activeOptionId}
          className="text-base-content placeholder:text-base-content/40 min-w-32 flex-1 bg-transparent outline-none"
          placeholder={single && values.length > 0 ? "" : placeholder}
          value={query}
          onChange={(event) => {
            const next = event.target.value;
            setQuery(next);
            setOpen(true);
            setFailed(false);
            setLoading(next.trim() !== "");
          }}
          onFocus={() => {
            if (query.trim() !== "") setOpen(true);
          }}
          onKeyDown={onKeyDown}
        />
      </div>

      {showDropdown && (
        <ul
          id={listboxId}
          role="listbox"
          className="border-base-300 bg-base-100 rounded-box absolute top-full left-0 z-30 mt-1 max-h-64 w-full overflow-auto border p-1 shadow-lg"
        >
          {loading ? (
            <li className="text-base-content/60 px-3 py-2 text-sm">
              Searching…
            </li>
          ) : failed ? (
            <li className="text-error px-3 py-2 text-sm">
              Search is unavailable. Check your connection and try again.
            </li>
          ) : visible.length === 0 ? (
            <li className="text-base-content/60 px-3 py-2 text-sm">
              No matches.
            </li>
          ) : (
            visible.map((result, index) => (
              <li
                key={result.value}
                id={`${listboxId}-opt-${index}`}
                role="option"
                aria-selected={index === highlight}
              >
                <button
                  type="button"
                  className={`flex w-full cursor-pointer items-center gap-2 rounded-md px-3 py-2 text-left text-sm ${
                    index === highlight ? "bg-base-200" : "hover:bg-base-200"
                  }`}
                  // Mouse-down (before the input blurs) so the pick lands.
                  onMouseDown={(event) => {
                    event.preventDefault();
                    addTerm(result.value);
                  }}
                  onMouseEnter={() => setHighlight(index)}
                >
                  <span className="min-w-0 flex-1 truncate">
                    {result.kind === "ticker"
                      ? (result.name ?? result.ticker ?? result.value)
                      : result.label}
                  </span>
                  {result.kind === "ticker" && result.ticker && result.name ? (
                    <span className="badge badge-ghost badge-sm shrink-0 font-mono">
                      {result.ticker}
                    </span>
                  ) : result.kind === "sic" ? (
                    <span className="badge badge-ghost badge-sm shrink-0">
                      Industry
                    </span>
                  ) : null}
                </button>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}
