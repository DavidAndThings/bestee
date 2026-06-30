import { useEffect, useMemo, useState } from "react";
import { jobsApi } from "../services/jobsApi";
import type { SicCode, SicTicker } from "../lib/types";

type ListState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "ready"; codes: SicCode[] };

type TickersState =
  | { status: "loading" }
  | { status: "error" }
  | { status: "ready"; tickers: SicTicker[] };

// SEC publishes ~1,000 SIC codes; page through them rather than painting the
// whole (filtered) list at once.
const PAGE_SIZE = 25;

/**
 * Browse the SEC SIC industry codes. Filter by code or title, and click a row
 * to see the ticker symbols classified under that code (fetched on demand).
 */
export default function SicCodesPage() {
  const [list, setList] = useState<ListState>({ status: "loading" });
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<SicCode | null>(null);
  const [tickers, setTickers] = useState<TickersState>({ status: "loading" });

  // Load the SIC list once on mount.
  useEffect(() => {
    let cancelled = false;
    jobsApi
      .listSicCodes()
      .then((codes) => {
        if (!cancelled) setList({ status: "ready", codes });
      })
      .catch(() => {
        if (!cancelled) setList({ status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Fetch tickers whenever a code is selected (loading state is set by the
  // click handler so nothing is set synchronously inside this effect).
  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    jobsApi
      .tickersForSic(selected.sicCode)
      .then((result) => {
        if (!cancelled) setTickers({ status: "ready", tickers: result });
      })
      .catch(() => {
        if (!cancelled) setTickers({ status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [selected]);

  const filtered = useMemo(() => {
    if (list.status !== "ready") return [];
    const needle = filter.trim().toLowerCase();
    if (!needle) return list.codes;
    return list.codes.filter(
      (code) =>
        code.sicCode.toLowerCase().includes(needle) ||
        code.industryTitle.toLowerCase().includes(needle),
    );
  }, [list, filter]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const start = safePage * PAGE_SIZE;
  const pageItems = filtered.slice(start, start + PAGE_SIZE);

  const openTickers = (code: SicCode) => {
    setSelected(code);
    setTickers({ status: "loading" });
  };

  return (
    <div className="p-4 sm:p-8">
      <div className="mx-auto max-w-4xl">
        <div className="mb-6 px-1">
          <h1 className="text-2xl font-semibold sm:text-3xl">
            SIC Code Directory
          </h1>
          <p className="text-base-content/70 mt-2 max-w-2xl">
            Standard Industrial Classification codes and their industry titles.
            Click a row to see the tickers classified under that code.
          </p>
        </div>

        <input
          type="text"
          autoComplete="off"
          className="input input-bordered mb-4 w-full"
          placeholder="Filter by code or industry…"
          value={filter}
          onChange={(event) => {
            setFilter(event.target.value);
            setPage(0);
          }}
        />

        {list.status === "loading" ? (
          <div className="flex justify-center p-12">
            <span className="loading loading-spinner loading-lg" />
          </div>
        ) : list.status === "error" ? (
          <div className="alert alert-error">
            <span>Couldn't load the SIC code list. Please try again.</span>
          </div>
        ) : filtered.length === 0 ? (
          <div className="border-base-300 rounded-box border border-dashed p-10 text-center">
            <p className="text-base-content/60">No codes match your filter.</p>
          </div>
        ) : (
          <>
            <div className="border-base-300 rounded-box max-h-[60vh] overflow-auto border">
              <table className="table-pin-rows table">
                <thead>
                  <tr>
                    <th>SIC Code</th>
                    <th>Industry Title</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {pageItems.map((code) => (
                    <tr
                      key={code.sicCode}
                      className="hover cursor-pointer"
                      onClick={() => openTickers(code)}
                    >
                      <td className="font-mono">{code.sicCode}</td>
                      <td>{code.industryTitle}</td>
                      <td className="text-right">
                        <button
                          type="button"
                          className="btn btn-ghost btn-xs"
                          onClick={(event) => {
                            event.stopPropagation();
                            openTickers(code);
                          }}
                        >
                          View tickers
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
              <span className="text-base-content/50 text-sm">
                Showing {start + 1}–{start + pageItems.length} of{" "}
                {filtered.length} · Page {safePage + 1} of {pageCount}
              </span>
              <div className="join">
                <button
                  type="button"
                  className="join-item btn btn-sm"
                  disabled={safePage === 0}
                  onClick={() => setPage(safePage - 1)}
                >
                  « Prev
                </button>
                <button
                  type="button"
                  className="join-item btn btn-sm"
                  disabled={safePage >= pageCount - 1}
                  onClick={() => setPage(safePage + 1)}
                >
                  Next »
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {selected && (
        <div className="modal modal-open" role="dialog">
          <div className="modal-box">
            <h3 className="wrap-break-word text-lg font-semibold">
              {selected.industryTitle}
            </h3>
            <p className="text-base-content/60 font-mono text-sm">
              SIC {selected.sicCode}
            </p>

            <div className="mt-4">
              {tickers.status === "loading" ? (
                <div className="flex justify-center p-6">
                  <span className="loading loading-spinner" />
                </div>
              ) : tickers.status === "error" ? (
                <p className="text-error">
                  Couldn't load tickers. Please try again.
                </p>
              ) : tickers.tickers.length === 0 ? (
                <p className="text-base-content/60">
                  No tickers are classified under this code.
                </p>
              ) : (
                <ul className="divide-base-300 max-h-[50vh] divide-y overflow-auto">
                  {tickers.tickers.map((entry) => (
                    <li
                      key={entry.ticker}
                      className="flex items-baseline gap-3 py-2"
                    >
                      <span className="badge badge-primary badge-outline shrink-0 font-mono">
                        {entry.ticker}
                      </span>
                      <span className="text-base-content/70 truncate text-sm">
                        {entry.name ?? "\u2014"}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="modal-action">
              <button
                type="button"
                className="btn"
                onClick={() => setSelected(null)}
              >
                Close
              </button>
            </div>
          </div>
          <button
            type="button"
            className="modal-backdrop"
            aria-label="Close"
            onClick={() => setSelected(null)}
          >
            close
          </button>
        </div>
      )}
    </div>
  );
}
