import type {
  Job,
  JobDetail,
  JobStatus,
  ResultTable,
  SearchResult,
  SicCode,
  SicTicker,
} from "../lib/types";

/**
 * A stand-in for a real backend. Jobs are persisted to localStorage, namespaced
 * per Clerk user id, and every operation simulates network latency. Swap this
 * file's internals for real HTTP calls without touching the page/hooks (see
 * `jobsApi.ts`, the single integration point).
 */

const JOBS_PREFIX = "bestee:jobs:";

function delay<T>(value: T, ms = 250): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

function jobsKey(userId: string): string {
  return `${JOBS_PREFIX}${userId}`;
}

function readJobs(userId: string): Job[] {
  try {
    const raw = localStorage.getItem(jobsKey(userId));
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as Job[]) : [];
  } catch {
    return [];
  }
}

function writeJobs(userId: string, jobs: Job[]): void {
  localStorage.setItem(jobsKey(userId), JSON.stringify(jobs));
}

/** schemaId -> result-id prefix the backend uses (matches the api `_LOADERS`). */
const RESULT_PREFIX: Record<string, string> = {
  "relative-rotation-graph": "rrg",
  "spectral-clustering": "clustering",
  "regime-detection": "regime",
  "fama-french": "fama_french",
};

/** A plausible deterministic-looking result id, e.g. `clustering_a1b2c3d4e5f60718`. */
function mockResultId(schemaId: string): string {
  const prefix = RESULT_PREFIX[schemaId] ?? "result";
  const hex = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
  return `${prefix}_${hex}`;
}

/** Mock data: a handful of previously submitted jobs across tools/statuses. */
function createSeedJobs(userId: string): Job[] {
  const now = Date.now();
  const minute = 60_000;
  const hour = 60 * minute;
  const day = 24 * hour;

  const make = (
    schemaId: string,
    status: Job["status"],
    ago: number,
    payload: Record<string, unknown>,
  ): Job => ({
    id: crypto.randomUUID(),
    userId,
    schemaId,
    status,
    payload,
    resultId: mockResultId(schemaId),
    createdAt: now - ago,
    updatedAt: now - ago,
  });

  return [
    make("relative-rotation-graph", "completed", 2 * day, {
      securities: ["AAPL", "MSFT", "GOOG"],
      lookback_window: 14,
      lookback_period: "1w",
      smoothing_method: "z-score",
      smoothing_window: 5,
    }),
    make("portfolio-backtest", "completed", 5 * hour, {
      portfolio_name: "Tech Core",
      securities: ["AAPL", "NVDA"],
      initial_capital: 100000,
      rebalance_frequency: "monthly",
      lookback_window: 60,
    }),
    make("relative-rotation-graph", "failed", 1 * day, {
      securities: ["TSLA", "RIVN"],
      lookback_window: 30,
      lookback_period: "1d",
      smoothing_method: "double-ema",
      smoothing_window: 10,
    }),
    make("portfolio-backtest", "running", 20 * minute, {
      portfolio_name: "Dividend Mix",
      securities: ["JNJ", "KO", "PG"],
      initial_capital: 50000,
      rebalance_frequency: "quarterly",
      lookback_window: 90,
    }),
    make("relative-rotation-graph", "queued", 3 * minute, {
      securities: ["SPY", "QQQ", "IWM"],
      lookback_window: 21,
      lookback_period: "1w",
      smoothing_method: "z-score",
      smoothing_window: 7,
    }),
  ];
}

function ensureSeeded(userId: string): Job[] {
  const existing = readJobs(userId);
  if (existing.length > 0) return existing;
  const seeded = createSeedJobs(userId);
  writeJobs(userId, seeded);
  return seeded;
}

export function listJobs(userId: string): Promise<Job[]> {
  const jobs = ensureSeeded(userId)
    .slice()
    .sort((a, b) => b.createdAt - a.createdAt);
  return delay(jobs);
}

export type SubmitJobInput = {
  schemaId: string;
  payload: Record<string, unknown>;
};

export type SubmitJobResult = {
  ok: boolean;
  requestId: string;
  /** The deterministic result id, known at submit time. */
  resultId: string;
  receivedAt: number;
};

/**
 * Pretends to dispatch a chart configuration payload to a backend, and records
 * it as a tracked job (uniquely identified by `requestId`).
 */
export function submitJob(
  userId: string,
  input: SubmitJobInput,
): Promise<SubmitJobResult> {
  const requestId = crypto.randomUUID();
  const resultId = mockResultId(input.schemaId);
  const now = Date.now();
  const job: Job = {
    id: requestId,
    userId,
    schemaId: input.schemaId,
    status: "queued",
    payload: input.payload,
    resultId,
    createdAt: now,
    updatedAt: now,
  };
  const jobs = ensureSeeded(userId);
  jobs.push(job);
  writeJobs(userId, jobs);

  console.info("[mockBackend] submit job", {
    userId,
    schemaId: input.schemaId,
    requestId,
    resultId,
    payload: input.payload,
  });
  return delay({ ok: true, requestId, resultId, receivedAt: now }, 900);
}

const STATE_FOR_STATUS: Record<JobStatus, string> = {
  queued: "PENDING",
  running: "STARTED",
  completed: "SUCCESS",
  failed: "FAILURE",
};

/** Find a stored job by id across every user's bucket (mock has no auth scope). */
function findJob(taskId: string): Job | null {
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key || !key.startsWith(JOBS_PREFIX)) continue;
    try {
      const parsed = JSON.parse(localStorage.getItem(key) ?? "[]") as Job[];
      const found = Array.isArray(parsed)
        ? parsed.find((job) => job.id === taskId)
        : undefined;
      if (found) return found;
    } catch {
      // Ignore unparseable buckets.
    }
  }
  return null;
}

/** Return one job's full record (its log), derived from the stored mock job. */
export function getJob(taskId: string): Promise<JobDetail> {
  const job = findJob(taskId);
  const done = job?.status === "completed" || job?.status === "failed";
  const detail: JobDetail = {
    taskId,
    schemaId: job?.schemaId ?? "",
    state: job ? STATE_FOR_STATUS[job.status] : "PENDING",
    status: job?.status ?? "queued",
    payload: job?.payload ?? null,
    resultId: job?.resultId ?? null,
    createdAt: job ? new Date(job.createdAt).toISOString() : null,
    startedAt: job ? new Date(job.createdAt + 1500).toISOString() : null,
    finishedAt: job && done ? new Date(job.updatedAt).toISOString() : null,
    elapsedSeconds:
      job && done ? Math.max(0, (job.updatedAt - job.createdAt) / 1000) : null,
    error:
      job?.status === "failed" ? "The job failed during processing." : null,
    traceback:
      job?.status === "failed"
        ? "Traceback (most recent call last):\n  RuntimeError: simulated failure"
        : null,
  };
  return delay(detail, 200);
}

/** Canned aspect tables so the Job Status "view table" links work in mock mode. */
const MOCK_ASPECT_TABLES: Record<
  string,
  { columns: string[]; rows: Record<string, unknown>[] }
> = {
  cluster_label: {
    columns: ["ticker", "cluster_label"],
    rows: [
      { ticker: "AAPL", cluster_label: 0 },
      { ticker: "MSFT", cluster_label: 0 },
      { ticker: "XOM", cluster_label: 1 },
    ],
  },
  coordinates: {
    columns: ["timestamp", "ticker", "relative_strength", "relative_momentum"],
    rows: [
      {
        timestamp: "2024-01-02 00:00:00",
        ticker: "AAPL",
        relative_strength: 1.04,
        relative_momentum: 0.21,
      },
      {
        timestamp: "2024-01-02 00:00:00",
        ticker: "MSFT",
        relative_strength: 0.98,
        relative_momentum: -0.12,
      },
    ],
  },
  regime_label: {
    columns: ["timestamp", "ticker", "regime_label"],
    rows: [
      { timestamp: "2024-01-02 00:00:00", ticker: "AAPL", regime_label: 0 },
      { timestamp: "2024-01-03 00:00:00", ticker: "AAPL", regime_label: 1 },
    ],
  },
  ff_residuals: {
    columns: ["date", "ticker", "specification", "residual", "p_value"],
    rows: [
      {
        date: "2024-01-15",
        ticker: "AAPL",
        specification: "FF6_2020-01-01_2023-12-31",
        residual: 0.012,
        p_value: 0.31,
      },
    ],
  },
};

/** Return a canned result-aspect table (mock stand-in for `GET /results/...`). */
export function getResultAspect(
  resultId: string,
  aspect: string,
): Promise<ResultTable> {
  const table = MOCK_ASPECT_TABLES[aspect] ?? { columns: [], rows: [] };
  return delay(
    { resultId, aspect, columns: table.columns, rows: table.rows },
    200,
  );
}

/**
 * A representative slice of the backend `/search` catalog: company names mixed
 * with SIC industry titles (which the SEC publishes upper-cased). Lets the
 * ticker autocomplete behave realistically against the mock. The real backend
 * resolves a chosen term to ticker symbols at submission time.
 */
/** A catalog entry plus the lowercased fields the mock search matches on. */
type MockEntry = SearchResult & { haystacks: string[] };

// Securities as [symbol, name] (stocks plus a couple of ETFs), matched by
// either field -- mirrors the live catalog built from the ticker universe.
const MOCK_SECURITIES: [string, string][] = [
  ["AAPL", "Apple Inc."],
  ["MSFT", "Microsoft Corporation"],
  ["NVDA", "NVIDIA Corporation"],
  ["GOOGL", "Alphabet Inc. Class A"],
  ["GOOG", "Alphabet Inc. Class C"],
  ["AMZN", "Amazon.com, Inc."],
  ["META", "Meta Platforms, Inc."],
  ["TSLA", "Tesla, Inc."],
  ["AVGO", "Broadcom Inc."],
  ["BRK.B", "Berkshire Hathaway Inc."],
  ["JPM", "JPMorgan Chase & Co."],
  ["V", "Visa Inc."],
  ["MA", "Mastercard Incorporated"],
  ["LLY", "Eli Lilly and Company"],
  ["UNH", "UnitedHealth Group Incorporated"],
  ["XOM", "Exxon Mobil Corporation"],
  ["JNJ", "Johnson & Johnson"],
  ["PG", "Procter & Gamble Company"],
  ["HD", "Home Depot, Inc."],
  ["BAC", "Bank of America Corporation"],
  ["ABBV", "AbbVie Inc."],
  ["KO", "Coca-Cola Company"],
  ["PEP", "PepsiCo, Inc."],
  ["COST", "Costco Wholesale Corporation"],
  ["ADBE", "Adobe Inc."],
  ["CRM", "Salesforce, Inc."],
  ["AMD", "Advanced Micro Devices, Inc."],
  ["NFLX", "Netflix, Inc."],
  ["INTC", "Intel Corporation"],
  ["CSCO", "Cisco Systems, Inc."],
  ["WMT", "Walmart Inc."],
  ["DIS", "Walt Disney Company"],
  ["MCD", "McDonald's Corporation"],
  ["NKE", "Nike, Inc."],
  ["ORCL", "Oracle Corporation"],
  ["QCOM", "QUALCOMM Incorporated"],
  ["TXN", "Texas Instruments Incorporated"],
  ["PFE", "Pfizer Inc."],
  ["CVX", "Chevron Corporation"],
  ["WFC", "Wells Fargo & Company"],
  ["GS", "Goldman Sachs Group, Inc."],
  ["MS", "Morgan Stanley"],
  ["IBM", "International Business Machines Corporation"],
  ["AXP", "American Express Company"],
  ["BA", "Boeing Company"],
  ["CAT", "Caterpillar Inc."],
  ["SBUX", "Starbucks Corporation"],
  ["MU", "Micron Technology, Inc."],
  ["PLTR", "Palantir Technologies Inc."],
  ["UBER", "Uber Technologies, Inc."],
  ["SPY", "SPDR S&P 500 ETF Trust"],
  ["QQQ", "Invesco QQQ Trust"],
];

// SIC industry titles (upper-case, as the SEC publishes them). A single term
// that expands to many tickers, so it stays one entry matched by its title.
const MOCK_SIC_TITLES: string[] = [
  "SERVICES-PREPACKAGED SOFTWARE",
  "SEMICONDUCTORS & RELATED DEVICES",
  "ELECTRONIC COMPUTERS",
  "NATIONAL COMMERCIAL BANKS",
  "STATE COMMERCIAL BANKS",
  "PHARMACEUTICAL PREPARATIONS",
  "BIOLOGICAL PRODUCTS (NO DIAGNOSTIC SUBSTANCES)",
  "CRUDE PETROLEUM & NATURAL GAS",
  "RETAIL-VARIETY STORES",
  "RETAIL-EATING PLACES",
  "MOTOR VEHICLES & PASSENGER CAR BODIES",
  "AIRCRAFT",
  "SERVICES-COMPUTER PROGRAMMING, DATA PROCESSING, ETC.",
  "TELEPHONE & TELEGRAPH APPARATUS",
  "BEVERAGES",
  "REAL ESTATE INVESTMENT TRUSTS",
  "SERVICES-COMPUTER INTEGRATED SYSTEMS DESIGN",
  "RETAIL-CATALOG & MAIL-ORDER HOUSES",
  "SERVICES-BUSINESS SERVICES, NEC",
];

const SEARCH_CATALOG: MockEntry[] = [
  ...MOCK_SECURITIES.map(([ticker, name]) => ({
    value: ticker,
    label: `${name} (${ticker})`,
    kind: "ticker" as const,
    ticker,
    name,
    haystacks: [ticker.toLowerCase(), name.toLowerCase()],
  })),
  ...MOCK_SIC_TITLES.map((title) => ({
    value: title,
    label: title,
    kind: "sic" as const,
    ticker: null,
    name: null,
    haystacks: [title.toLowerCase()],
  })),
];

/** A representative slice of the SEC SIC code list for the mock browser. */
const SIC_CODES: SicCode[] = [
  { sicCode: "1311", industryTitle: "CRUDE PETROLEUM & NATURAL GAS" },
  { sicCode: "2080", industryTitle: "BEVERAGES" },
  { sicCode: "2834", industryTitle: "PHARMACEUTICAL PREPARATIONS" },
  {
    sicCode: "2836",
    industryTitle: "BIOLOGICAL PRODUCTS (NO DIAGNOSTIC SUBSTANCES)",
  },
  { sicCode: "3571", industryTitle: "ELECTRONIC COMPUTERS" },
  { sicCode: "3674", industryTitle: "SEMICONDUCTORS & RELATED DEVICES" },
  { sicCode: "3711", industryTitle: "MOTOR VEHICLES & PASSENGER CAR BODIES" },
  { sicCode: "3721", industryTitle: "AIRCRAFT" },
  { sicCode: "3826", industryTitle: "LABORATORY ANALYTICAL INSTRUMENTS" },
  { sicCode: "5812", industryTitle: "RETAIL-EATING PLACES" },
  { sicCode: "5961", industryTitle: "RETAIL-CATALOG & MAIL-ORDER HOUSES" },
  { sicCode: "6021", industryTitle: "NATIONAL COMMERCIAL BANKS" },
  { sicCode: "6798", industryTitle: "REAL ESTATE INVESTMENT TRUSTS" },
  { sicCode: "7372", industryTitle: "SERVICES-PREPACKAGED SOFTWARE" },
  {
    sicCode: "7389",
    industryTitle: "SERVICES-COMPUTER PROGRAMMING, DATA PROCESSING, ETC.",
  },
];

/** Mock ticker membership per SIC code; unmapped codes return none. */
const SIC_TICKERS: Record<string, SicTicker[]> = {
  "1311": [
    { ticker: "XOM", name: "Exxon Mobil Corporation" },
    { ticker: "CVX", name: "Chevron Corporation" },
    { ticker: "COP", name: "ConocoPhillips" },
    { ticker: "OXY", name: "Occidental Petroleum Corporation" },
  ],
  "2080": [
    { ticker: "KO", name: "Coca-Cola Company" },
    { ticker: "PEP", name: "PepsiCo, Inc." },
    { ticker: "MNST", name: "Monster Beverage Corporation" },
  ],
  "2834": [
    { ticker: "PFE", name: "Pfizer Inc." },
    { ticker: "MRK", name: "Merck & Co., Inc." },
    { ticker: "LLY", name: "Eli Lilly and Company" },
    { ticker: "ABBV", name: "AbbVie Inc." },
    { ticker: "BMY", name: "Bristol-Myers Squibb Company" },
  ],
  "3571": [
    { ticker: "AAPL", name: "Apple Inc." },
    { ticker: "DELL", name: "Dell Technologies Inc." },
    { ticker: "HPQ", name: "HP Inc." },
  ],
  "3674": [
    { ticker: "NVDA", name: "NVIDIA Corporation" },
    { ticker: "AVGO", name: "Broadcom Inc." },
    { ticker: "AMD", name: "Advanced Micro Devices, Inc." },
    { ticker: "TXN", name: "Texas Instruments Incorporated" },
    { ticker: "MU", name: "Micron Technology, Inc." },
    { ticker: "QCOM", name: "QUALCOMM Incorporated" },
  ],
  "3711": [
    { ticker: "TSLA", name: "Tesla, Inc." },
    { ticker: "GM", name: "General Motors Company" },
    { ticker: "F", name: "Ford Motor Company" },
  ],
  "3721": [{ ticker: "BA", name: "Boeing Company" }],
  "5812": [
    { ticker: "MCD", name: "McDonald's Corporation" },
    { ticker: "SBUX", name: "Starbucks Corporation" },
    { ticker: "CMG", name: "Chipotle Mexican Grill, Inc." },
    { ticker: "YUM", name: "Yum! Brands, Inc." },
  ],
  "6021": [
    { ticker: "JPM", name: "JPMorgan Chase & Co." },
    { ticker: "BAC", name: "Bank of America Corporation" },
    { ticker: "WFC", name: "Wells Fargo & Company" },
    { ticker: "C", name: "Citigroup Inc." },
  ],
  "7372": [
    { ticker: "MSFT", name: "Microsoft Corporation" },
    { ticker: "ORCL", name: "Oracle Corporation" },
    { ticker: "ADBE", name: "Adobe Inc." },
    { ticker: "CRM", name: "Salesforce, Inc." },
    { ticker: "NOW", name: "ServiceNow, Inc." },
  ],
};

/** The full SIC code list (mocked). */
export function listSicCodes(): Promise<SicCode[]> {
  return delay(SIC_CODES.slice(), 200);
}

/** Tickers (with company names) classified under *sicCode* (mocked). */
export function tickersForSic(sicCode: string): Promise<SicTicker[]> {
  return delay((SIC_TICKERS[sicCode] ?? []).slice(), 200);
}

/**
 * Search the catalog for *query* by ticker symbol, company/ETF name, or SIC
 * industry title (case-insensitive). Exact matches rank ahead of prefixes,
 * which rank ahead of other substrings -- mirroring the backend `/search`.
 * Swap this for an authenticated HTTP call when wiring up the live API.
 */
export function search(query: string, limit = 20): Promise<SearchResult[]> {
  const needle = query.trim().toLowerCase();
  if (!needle) return delay<SearchResult[]>([], 120);
  const ranked: { tier: number; label: string; entry: MockEntry }[] = [];
  for (const entry of SEARCH_CATALOG) {
    let tier: number | null = null;
    for (const field of entry.haystacks) {
      if (!field.includes(needle)) continue;
      const t = field === needle ? 0 : field.startsWith(needle) ? 1 : 2;
      tier = tier === null ? t : Math.min(tier, t);
    }
    if (tier !== null) {
      ranked.push({ tier, label: entry.label.toLowerCase(), entry });
    }
  }
  ranked.sort((a, b) => a.tier - b.tier || a.label.localeCompare(b.label));
  const out: SearchResult[] = [];
  const seen = new Set<string>();
  for (const { entry } of ranked) {
    if (seen.has(entry.value)) continue;
    seen.add(entry.value);
    out.push({
      value: entry.value,
      label: entry.label,
      kind: entry.kind,
      ticker: entry.ticker ?? null,
      name: entry.name ?? null,
    });
    if (out.length >= limit) break;
  }
  return delay(out, 120);
}
