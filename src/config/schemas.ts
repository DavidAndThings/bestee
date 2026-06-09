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

export type FieldType = "string" | "integer" | "array";

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

const PORTFOLIO_BACKTEST_SCHEMA: Schema = {
  id: "portfolio-backtest",
  name: "Portfolio Backtest",
  description:
    "Backtest a weighted portfolio of securities over historical data.",
  badge: "Portfolio",
  imageUrl: paperMoney,
  buttonVariant: "btn-secondary",
  parameters: {
    portfolio_name: {
      type: "string",
      description: "A name for this backtest.",
    },
    securities: {
      type: "array",
      items: {
        type: "string",
      },
      description: "The securities to include in the portfolio.",
    },
    initial_capital: {
      type: "integer",
      description: "The starting capital for the backtest, in whole dollars.",
    },
    rebalance_frequency: {
      type: "string",
      description:
        "How often to rebalance the portfolio back to target weights.",
      choices: ["daily", "weekly", "monthly", "quarterly"],
    },
    lookback_window: {
      type: "integer",
      description: "How many periods of history to use when sizing positions.",
    },
  },
};

/**
 * The single source of truth for available charts. Each entry renders a card on
 * the home page and powers a chart setup form at `/charts/:id`. Add a new chart
 * by appending one `Schema` object here.
 */
export const SCHEMA_REGISTRY: Schema[] = [
  RELATIVE_ROTATION_GRAPH_SCHEMA,
  PORTFOLIO_BACKTEST_SCHEMA,
];

export function getSchema(id: string | undefined): Schema | undefined {
  if (!id) return undefined;
  return SCHEMA_REGISTRY.find((schema) => schema.id === id);
}

export { RELATIVE_ROTATION_GRAPH_SCHEMA, PORTFOLIO_BACKTEST_SCHEMA };
