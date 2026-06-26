import barChart from "../assets/gifs/bar-chart.gif";
import paperMoney from "../assets/gifs/paper-money.gif";

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
};

const RELATIVE_ROTATION_GRAPH_SCHEMA: Schema = {
  id: "relative-rotation-graph",
  name: "Relative Rotation Graph",
  description: "The relative rotation graph for a given set of securities.",
  badge: "Markets",
  imageUrl: barChart,
  buttonVariant: "btn-primary",
  parameters: {
    securities: {
      type: "array",
      items: {
        type: "string",
      },
      description: "The securities to include in the graph.",
    },
    lookback_window: {
      type: "integer",
      description: "How many periods to look back for each security.",
    },
    lookback_period: {
      type: "string",
      description:
        "The period to use for the lookback (e.g. '1d', '1w', '1m').",
      choices: ["1d", "1w", "1m"],
    },
    smoothing_method: {
      type: "string",
      description:
        "The smoothing method to use for Relative Strength and Relative Momentum calculations.",
      choices: ["z-score", "double-ema"],
    },
    smoothing_window: {
      type: "integer",
      description: "The window size for timer series smoothing.",
    },
  },
};

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

/**
 * The single source of truth for available charts. Each entry renders a card on
 * the home page and powers a chart setup form at `/charts/:id`. Add a new chart
 * by appending one `Schema` object here.
 */
export const SCHEMA_REGISTRY: Schema[] = [
  RELATIVE_ROTATION_GRAPH_SCHEMA,
  TRAINING_DATA_SELECTION_SCHEMA,
];

export function getSchema(id: string | undefined): Schema | undefined {
  if (!id) return undefined;
  return SCHEMA_REGISTRY.find((schema) => schema.id === id);
}

export { RELATIVE_ROTATION_GRAPH_SCHEMA, TRAINING_DATA_SELECTION_SCHEMA };
