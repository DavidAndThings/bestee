export type ChatRole = "user" | "assistant" | "system";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
};

/** A scalar value type that a single user answer can be parsed into. */
export type ScalarType = "string" | "integer";

/**
 * A question derived from a schema field. For array fields, `valueType`
 * describes the type of each item and `isArray` is true.
 */
export type Question = {
  fieldKey: string;
  prompt: string;
  valueType: ScalarType;
  isArray: boolean;
  choices?: string[];
};

/**
 * A completed function call the agent wants to make, awaiting user
 * confirmation before it is dispatched to the backend.
 */
export type PendingToolCall = {
  schemaId: string;
  arguments: Record<string, unknown>;
};

/**
 * A single, continuous agent conversation per user. The assistant can fill and
 * submit any tool (schema) and switch tools mid-conversation, so there is just
 * one persisted conversation rather than one thread per tool. This shape is
 * persisted by the (mock) backend, so it must stay JSON-serialisable.
 */
export type Conversation = {
  id: string;
  /** Clerk user id. The conversation is isolated per user. */
  userId: string;
  messages: ChatMessage[];
  /** The tool the agent is currently helping with, if any. */
  activeSchemaId: string | null;
  /** Arguments gathered for the active tool so far, keyed by field name. */
  collected: Record<string, unknown>;
  /** The field the agent last asked about (drives choice chips and parsing). */
  pendingField: string | null;
  /** Set when all args are gathered and we're awaiting the user's confirmation. */
  pendingToolCall: PendingToolCall | null;
  createdAt: number;
  updatedAt: number;
};
