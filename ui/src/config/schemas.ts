import server from "../assets/gifs/isometric-server.gif";
import nokia from "../assets/gifs/retro-nokia.gif";
import computer from "../assets/gifs/computer.gif";
import paperMoney from "../assets/gifs/paper-money.gif";
import pencil from "../assets/gifs/pencil.gif";

export type ButtonVariant =
  | "btn-primary"
  | "btn-secondary"
  | "btn-accent"
  | "btn-info"
  | "btn-success"
  | "btn-warning"
  | "btn-error";

export type FieldType = "string" | "integer" | "date" | "array";

export type FieldDef = {
  type: FieldType;
  description: string;
  choices?: string[];
  items?: { type: FieldType };
  /** Whether the field may be omitted (mirrors an optional / defaulted model field). */
  optional?: boolean;
  /** Default pre-fill value (mirrors the model field's default). */
  default?: string | number;
  /**
   * Render with the security search autocomplete: the user picks terms (ticker
   * symbols, company names, or SIC industry titles) that the backend resolves
   * to ticker symbols. Use on an `array` field for multi-select, or a `string`
   * field for a single security (e.g. an RRG benchmark).
   */
  ticker?: boolean;
};

export type Schema = {
  /** Stable identifier used as the URL param and storage key. */
  id: string;
  /** Card title and chart setup heading. */
  name: string;
  /** Card description and chart setup intro. */
  description: string;
  badge?: string;
  imageUrl?: string;
  buttonVariant?: ButtonVariant;
  /**
   * Ordered map of field definitions. Insertion order determines the order in
   * which fields are rendered in the chart setup form.
   */
  parameters: Record<string, FieldDef>;
  /**
   * Optional cross-field validation, run only once every field passes its own
   * checks. Returns a map of field key -> error message (empty when valid).
   */
  validate?: (payload: Record<string, unknown>) => Record<string, string>;
  /**
   * Optional mapping from the validated form payload to the request body the
   * API expects, for schemas whose form fields don't map 1:1 to the backend
   * model (e.g. Fama-French collects a single window + OOS date but the model
   * takes `intervals` + `oos_dates`). Defaults to the payload unchanged.
   */
  toRequestBody?: (payload: Record<string, unknown>) => Record<string, unknown>;
};

const RELATIVE_ROTATION_GRAPH_SCHEMA: Schema = {
  id: "relative-rotation-graph",
  name: "Relative Rotation Graph",
  description: "The relative rotation graph for a given set of securities.",
  badge: "Markets",
  imageUrl: server,
  buttonVariant: "btn-primary",
  parameters: {
    tickers: {
      type: "array",
      items: {
        type: "string",
      },
      ticker: true,
      description: "The securities to plot on the rotation graph.",
    },
    start_date: {
      type: "date",
      description: "Start of the analysis window.",
    },
    end_date: {
      type: "date",
      description: "End of the analysis window.",
    },
    reference_type: {
      type: "string",
      description:
        "Measure relative strength against an equal-weight basket of the tickers ('mean') or a single benchmark security ('ticker').",
      choices: ["mean", "ticker"],
      default: "mean",
    },
    benchmark_ticker: {
      type: "string",
      ticker: true,
      description:
        "Benchmark security to rotate against. Required when reference_type is 'ticker'.",
      optional: true,
    },
  },
  validate: (payload) => {
    const errors: Record<string, string> = {};
    const start = payload.start_date;
    const end = payload.end_date;
    if (typeof start === "string" && typeof end === "string" && start > end) {
      errors.end_date = "The end date must be on or after the start date.";
    }
    if (payload.reference_type === "ticker" && !payload.benchmark_ticker) {
      errors.benchmark_ticker =
        "A benchmark ticker is required when reference type is 'ticker'.";
    }
    return errors;
  },
};

/**
 * @deprecated The training-data-selection workflow (agglomerative clustering /
 * representative selection) was retired from the backend. It is no longer listed
 * in `SCHEMA_REGISTRY`; the export is kept only for reference and old-job lookups.
 */
const TRAINING_DATA_SELECTION_SCHEMA: Schema = {
  id: "training-data-selection",
  name: "Training Data Selection",
  description:
    "Select the training data to use for model training. Currently this process is optimized for training on a random forest trading model.",
  badge: "Training",
  imageUrl: paperMoney,
  buttonVariant: "btn-secondary",
  parameters: {
    lookback_window_start: {
      type: "date",
      description: "The start date of the lookback window.",
    },
    lookback_window_end: {
      type: "date",
      description: "The end date of the lookback window.",
    },
    lookback_period: {
      type: "string",
      description: "The period to use for lookback.",
      choices: ["daily", "weekly", "monthly"],
    },
    distance_metric: {
      type: "string",
      description:
        "The metric to use to quantify the distance between two securities.",
      choices: [
        "Positive Semi-Correlation",
        "Negative Semi-Correlation",
        "Mutual Information",
        "Tail Dependence",
      ],
    },
    linkage_method: {
      type: "string",
      description: "The linkage method to use for agglomerative clustering.",
      choices: ["ward", "complete", "average", "single"],
    },
    cluster_selection_method: {
      type: "string",
      description:
        "The method to use for selecting clusters after hierarchical clustering.",
      choices: [
        "Maximize Silhouette Score Optimization",
        "Optimal Number of Clusters (ONC)",
        "Gap Statistic",
        "Clest",
      ],
    },
    representative_selection_method: {
      type: "string",
      description:
        "The representative selection method to use for selecting a representative security from each cluster.",
      choices: [
        "Medoid",
        "Max-Sharpe",
        "Inverse-Variance Centralizer",
        "Anti-Medoid",
        "Highest Historic Beta",
      ],
    },
  },
  validate: (payload) => {
    const errors: Record<string, string> = {};
    const start = payload.lookback_window_start;
    const end = payload.lookback_window_end;
    // ISO yyyy-mm-dd strings order chronologically as plain strings.
    if (typeof start === "string" && typeof end === "string" && start > end) {
      errors.lookback_window_end =
        "The end date must be on or after the start date.";
    }
    return errors;
  },
};

const SPECTRAL_CLUSTERING_SCHEMA: Schema = {
  id: "spectral-clustering",
  name: "Spectral Clustering",
  description:
    "Group securities into clusters by their residual (market-neutral) co-movement.",
  badge: "Clustering",
  imageUrl: computer,
  buttonVariant: "btn-accent",
  parameters: {
    tickers: {
      type: "array",
      items: {
        type: "string",
      },
      ticker: true,
      description: "The securities to cluster.",
    },
    start_date: {
      type: "date",
      description: "Start of the analysis window.",
    },
    end_date: {
      type: "date",
      description: "End of the analysis window.",
    },
    min_num_clusters: {
      type: "integer",
      description:
        "Smallest number of clusters to consider; the best count is selected within this range.",
      default: 2,
    },
    max_num_clusters: {
      type: "integer",
      description:
        "Largest number of clusters to consider; the best count is selected within this range.",
      default: 50,
    },
  },
  validate: (payload) => {
    const errors: Record<string, string> = {};
    const start = payload.start_date;
    const end = payload.end_date;
    if (typeof start === "string" && typeof end === "string" && start > end) {
      errors.end_date = "The end date must be on or after the start date.";
    }
    const min = payload.min_num_clusters;
    const max = payload.max_num_clusters;
    if (typeof min === "number" && typeof max === "number" && min > max) {
      errors.max_num_clusters =
        "The maximum number of clusters must be at least the minimum.";
    }
    return errors;
  },
};

const REGIME_DETECTION_SCHEMA: Schema = {
  id: "regime-detection",
  name: "Regime Detection",
  description:
    "Label each security's history into market regimes (e.g. calm vs. turbulent).",
  badge: "Markets",
  imageUrl: pencil,
  buttonVariant: "btn-info",
  parameters: {
    tickers: {
      type: "array",
      items: {
        type: "string",
      },
      ticker: true,
      description: "The securities to detect regimes for.",
    },
    start_date: {
      type: "date",
      description: "Start of the analysis window.",
    },
    end_date: {
      type: "date",
      description: "End of the analysis window.",
    },
    benchmark_ticker: {
      type: "string",
      ticker: true,
      description:
        "Optional benchmark security. When set, each asset is residualized against it with a rolling market-model (OLS) regression; left blank, residualization falls back to PCA over the basket itself.",
      optional: true,
    },
  },
  validate: (payload) => {
    const errors: Record<string, string> = {};
    const start = payload.start_date;
    const end = payload.end_date;
    if (typeof start === "string" && typeof end === "string" && start > end) {
      errors.end_date = "The end date must be on or after the start date.";
    }
    return errors;
  },
};

const FAMA_FRENCH_SCHEMA: Schema = {
  id: "fama-french",
  name: "Fama-French Factor Model",
  description:
    "Fit a Fama-French factor regression for each security over an estimation window.",
  badge: "Factors",
  imageUrl: nokia,
  buttonVariant: "btn-success",
  parameters: {
    tickers: {
      type: "array",
      items: {
        type: "string",
      },
      ticker: true,
      description: "The securities to fit factor models for.",
    },
    start_date: {
      type: "date",
      description: "Start of the estimation window.",
    },
    end_date: {
      type: "date",
      description: "End of the estimation window.",
    },
    oos_date: {
      type: "date",
      description:
        "Out-of-sample date for the residuals; must fall after the end date.",
    },
  },
  validate: (payload) => {
    const errors: Record<string, string> = {};
    const start = payload.start_date;
    const end = payload.end_date;
    const oos = payload.oos_date;
    if (typeof start === "string" && typeof end === "string" && start > end) {
      errors.end_date = "The end date must be on or after the start date.";
    }
    if (typeof end === "string" && typeof oos === "string" && oos <= end) {
      errors.oos_date = "The out-of-sample date must be after the end date.";
    }
    return errors;
  },
  // The model takes one or more estimation `intervals` and `oos_dates`; the
  // form collects a single window + OOS date, wrapped here into those lists.
  toRequestBody: (payload) => ({
    tickers: payload.tickers,
    intervals: [{ start_date: payload.start_date, end_date: payload.end_date }],
    oos_dates: [payload.oos_date],
  }),
};

/**
 * The single source of truth for available charts. Each entry renders a card on
 * the home page and powers a chart setup form at `/charts/:id`. Add a new chart
 * by appending one `Schema` object here.
 */
export const SCHEMA_REGISTRY: Schema[] = [
  RELATIVE_ROTATION_GRAPH_SCHEMA,
  SPECTRAL_CLUSTERING_SCHEMA,
  REGIME_DETECTION_SCHEMA,
  FAMA_FRENCH_SCHEMA,
];

export function getSchema(id: string | undefined): Schema | undefined {
  if (!id) return undefined;
  return SCHEMA_REGISTRY.find((schema) => schema.id === id);
}

export {
  FAMA_FRENCH_SCHEMA,
  REGIME_DETECTION_SCHEMA,
  RELATIVE_ROTATION_GRAPH_SCHEMA,
  SPECTRAL_CLUSTERING_SCHEMA,
  TRAINING_DATA_SELECTION_SCHEMA,
};
