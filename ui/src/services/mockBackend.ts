import type { Job, ResultTable, SicCode, SicTicker } from "../lib/types";

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
const SEARCH_CATALOG: string[] = [
  // Company names (as the Massive ticker universe reports them).
  "Apple Inc.",
  "Microsoft Corporation",
  "NVIDIA Corporation",
  "Alphabet Inc. Class A",
  "Alphabet Inc. Class C",
  "Amazon.com, Inc.",
  "Meta Platforms, Inc.",
  "Tesla, Inc.",
  "Broadcom Inc.",
  "Berkshire Hathaway Inc.",
  "JPMorgan Chase & Co.",
  "Visa Inc.",
  "Mastercard Incorporated",
  "Eli Lilly and Company",
  "UnitedHealth Group Incorporated",
  "Exxon Mobil Corporation",
  "Johnson & Johnson",
  "Procter & Gamble Company",
  "Home Depot, Inc.",
  "Bank of America Corporation",
  "AbbVie Inc.",
  "Coca-Cola Company",
  "PepsiCo, Inc.",
  "Costco Wholesale Corporation",
  "Adobe Inc.",
  "Salesforce, Inc.",
  "Advanced Micro Devices, Inc.",
  "Netflix, Inc.",
  "Intel Corporation",
  "Cisco Systems, Inc.",
  "Walmart Inc.",
  "Walt Disney Company",
  "McDonald's Corporation",
  "Nike, Inc.",
  "Oracle Corporation",
  "QUALCOMM Incorporated",
  "Texas Instruments Incorporated",
  "Pfizer Inc.",
  "Chevron Corporation",
  "Wells Fargo & Company",
  "Goldman Sachs Group, Inc.",
  "Morgan Stanley",
  "International Business Machines Corporation",
  "American Express Company",
  "Boeing Company",
  "Caterpillar Inc.",
  "Starbucks Corporation",
  "Micron Technology, Inc.",
  "Palantir Technologies Inc.",
  "Uber Technologies, Inc.",
  // SIC industry titles (upper-case, as the SEC publishes them).
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
 * Search the catalog for *query* (case-insensitive substring), ranking prefix
 * matches ahead of other matches -- mirroring the backend `/search` endpoint.
 * Swap this for an authenticated HTTP call when wiring up the live API.
 */
export function search(query: string, limit = 20): Promise<string[]> {
  const needle = query.trim().toLowerCase();
  if (!needle) return delay<string[]>([], 120);
  const prefix: string[] = [];
  const other: string[] = [];
  const seen = new Set<string>();
  for (const term of SEARCH_CATALOG) {
    const folded = term.toLowerCase();
    if (!folded.includes(needle) || seen.has(folded)) continue;
    seen.add(folded);
    (folded.startsWith(needle) ? prefix : other).push(term);
  }
  prefix.sort();
  other.sort();
  return delay([...prefix, ...other].slice(0, limit), 120);
}
