import type { FieldDef, Schema } from "../config/schemas";
import type { ChatMessage, PendingToolCall, Question } from "../lib/types";
import { schemaToQuestions } from "../lib/schemaToQuestions";
import { isDoneSignal, parseScalar } from "../lib/parseUserInput";
import { humanizeKey } from "../lib/format";

/**
 * ─────────────────────────────────────────────────────────────────────────
 *  MOCK LLM AGENT (function-calling style)
 * ─────────────────────────────────────────────────────────────────────────
 *
 * `runAgentTurn` is the single seam to swap for a real LLM. The schemas in
 * `SCHEMA_REGISTRY` are already shaped like function/tool definitions
 * (name, description, parameters with type/description/choices), so wiring a
 * real provider is mostly mapping:
 *
 *   1. Build `tools` from `input.schemas` (each schema -> one function).
 *   2. Build `messages` from `input.history` (+ a system prompt) and append
 *      the new user message.
 *   3. Call the provider with `tool_choice: "auto"` SERVER-SIDE (never expose
 *      the API key in the browser — proxy through your backend / chatApi).
 *   4. Map the response:
 *        - assistant text  -> `reply`
 *        - a tool_call      -> `toolCall: { schemaId, arguments }`
 *        - which tool it's filling -> `activeSchemaId`
 *        - partial arguments so far -> `collected`
 *
 * The mock below approximates that behaviour deterministically: it detects the
 * user's intended tool, fills one field per turn (plus light extraction), lets
 * the user switch tools mid-conversation, and emits a tool call once every
 * field is gathered.
 */

export type AgentInput = {
  schemas: Schema[];
  /** Prior working state. A real LLM can re-derive this from `history`. */
  activeSchemaId: string | null;
  collected: Record<string, unknown>;
  pendingField: string | null;
  /** Full transcript so far (what a real LLM needs). */
  history: ChatMessage[];
  /** The new user message to process this turn. */
  userMessage: string;
};

export type AgentOutput = {
  reply: string;
  activeSchemaId: string | null;
  collected: Record<string, unknown>;
  pendingField: string | null;
  /** Present when the agent is ready to run a tool (awaiting confirmation). */
  toolCall?: PendingToolCall;
};

function delay<T>(value: T, ms = 600): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

const STOP_WORDS = new Set([
  "the",
  "a",
  "an",
  "of",
  "for",
  "and",
  "to",
  "with",
  "please",
  "let",
  "lets",
  "build",
  "create",
  "run",
  "make",
  "do",
  "want",
  "like",
  "would",
  "i",
  "id",
  "my",
  "me",
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

/** Distinctive keywords for a tool, drawn from its name and id. */
function schemaKeywords(schema: Schema): string[] {
  const words = [...tokenize(schema.name), ...tokenize(schema.id)];
  return [...new Set(words)].filter((word) => !STOP_WORDS.has(word));
}

function getSchemaById(
  schemas: Schema[],
  id: string | null,
): Schema | undefined {
  if (!id) return undefined;
  return schemas.find((schema) => schema.id === id);
}

/** Scores how strongly a message refers to each tool; returns the best match. */
function detectSchema(
  schemas: Schema[],
  message: string,
): { schema: Schema; score: number } | null {
  const tokens = new Set(tokenize(message));
  const lower = message.toLowerCase();
  let best: { schema: Schema; score: number } | null = null;

  for (const schema of schemas) {
    let score = 0;
    for (const keyword of schemaKeywords(schema)) {
      if (tokens.has(keyword)) score += 1;
    }
    if (lower.includes(schema.name.toLowerCase())) score += 2;
    if (score > 0 && (!best || score > best.score)) {
      best = { schema, score };
    }
  }
  return best;
}

function promptForField(question: Question): string {
  if (question.isArray) {
    return `${question.prompt} (send one or more, then say "done")`;
  }
  if (question.choices && question.choices.length > 0) {
    return `${question.prompt} Options: ${question.choices.join(", ")}.`;
  }
  return question.prompt;
}

function toolMenu(schemas: Schema[]): string {
  const list = schemas.map((schema) => `• ${schema.name}`).join("\n");
  return `I can help you with:\n${list}\n\nWhich would you like to do?`;
}

/** Splits a free-form message into individual array items (e.g. tickers). */
function parseArrayItems(raw: string): string[] {
  return raw
    .split(/[,\n]|\band\b/i)
    .flatMap((part) => part.split(/\s+/))
    .map((token) => token.trim())
    .filter(
      (token) => token.length > 0 && !STOP_WORDS.has(token.toLowerCase()),
    );
}

function isMissing(
  question: Question,
  collected: Record<string, unknown>,
): boolean {
  const value = collected[question.fieldKey];
  if (value === undefined) return true;
  if (question.isArray) return !Array.isArray(value) || value.length === 0;
  return false;
}

/** A valid answer to the field currently being asked (used to avoid switching). */
function looksLikeAnswer(
  question: Question | undefined,
  message: string,
): boolean {
  if (!question) return false;
  if (question.isArray) return true;
  return parseScalar(question, message).ok;
}

export function runAgentTurn(input: AgentInput): Promise<AgentOutput> {
  const { schemas } = input;
  let activeSchemaId = input.activeSchemaId;
  let collected = { ...input.collected };
  let pendingField = input.pendingField;
  const message = input.userMessage.trim();

  const activeSchema = getSchemaById(schemas, activeSchemaId);
  const currentQuestion =
    activeSchema && pendingField
      ? schemaToQuestions(activeSchema).find((q) => q.fieldKey === pendingField)
      : undefined;

  // 1) Intent detection / tool switching.
  const match = detectSchema(schemas, message);
  const strongSignal = match != null && match.score >= 2;
  const wantsSwitch =
    match != null &&
    match.schema.id !== activeSchemaId &&
    (activeSchemaId == null ||
      strongSignal ||
      !looksLikeAnswer(currentQuestion, message));

  let prefix = "";
  if (wantsSwitch && match) {
    activeSchemaId = match.schema.id;
    collected = {};
    pendingField = null;
    prefix = `Sure — let's set up your ${match.schema.name}. `;
  }

  // 2) No tool yet: ask which one.
  if (!activeSchemaId) {
    return delay({
      reply: toolMenu(schemas),
      activeSchemaId: null,
      collected,
      pendingField: null,
    });
  }

  const schema = getSchemaById(schemas, activeSchemaId)!;
  const questions = schemaToQuestions(schema);
  const byKey = new Map(questions.map((q) => [q.fieldKey, q]));

  // 3) If we just asked for a field, treat this message as the answer.
  if (pendingField && !wantsSwitch) {
    const question = byKey.get(pendingField)!;

    if (question.isArray) {
      if (isDoneSignal(message)) {
        const arr = collected[pendingField];
        if (!Array.isArray(arr) || arr.length === 0) {
          return delay({
            reply: `Please add at least one ${humanizeKey(pendingField)} value before we continue.`,
            activeSchemaId,
            collected,
            pendingField,
          });
        }
        pendingField = null;
      } else {
        const items = parseArrayItems(message);
        if (items.length === 0) {
          return delay({
            reply: `I didn't catch any values there. ${promptForField(question)}`,
            activeSchemaId,
            collected,
            pendingField,
          });
        }
        const existing = Array.isArray(collected[pendingField])
          ? (collected[pendingField] as unknown[])
          : [];
        collected = { ...collected, [pendingField]: [...existing, ...items] };
        return delay({
          reply: `Added ${items.map((i) => `"${i}"`).join(", ")}. Add more, or say "done".`,
          activeSchemaId,
          collected,
          pendingField,
        });
      }
    } else {
      const result = parseScalar(question, message);
      if (!result.ok) {
        return delay({
          reply: result.error,
          activeSchemaId,
          collected,
          pendingField,
        });
      }
      collected = { ...collected, [pendingField]: result.value };
      pendingField = null;
    }
  }

  // 4) Light extraction when (re)entering a tool: fill any choice fields the
  //    user already mentioned. (A real LLM would extract all fields at once.)
  if (wantsSwitch) {
    const lower = message.toLowerCase();
    for (const question of questions) {
      if (!isMissing(question, collected) || !question.choices) continue;
      const found = question.choices.find((choice) =>
        lower.includes(choice.toLowerCase()),
      );
      if (found) collected = { ...collected, [question.fieldKey]: found };
    }
  }

  // 5) Ask for the next missing field, or emit a tool call.
  const next = questions.find((question) => isMissing(question, collected));
  if (next) {
    pendingField = next.fieldKey;
    return delay({
      reply: prefix + promptForField(next),
      activeSchemaId,
      collected,
      pendingField,
    });
  }

  return delay({
    reply:
      prefix +
      `Great — I have everything for your ${schema.name}. Review the details below and confirm to run it.`,
    activeSchemaId,
    collected,
    pendingField: null,
    toolCall: { schemaId: schema.id, arguments: { ...collected } },
  });
}

/** Exposed so the chat hook can phrase a re-ask when the user edits a field. */
export { promptForField };

// REFERENCE — swapping the mock for real OpenAI function calling:
//   1. Tools:    input.schemas.map(toOpenAITool)  (mapper defined below).
//   2. Call the model SERVER-SIDE with tool_choice "auto", passing a system
//      prompt + input.history + input.userMessage. Never put the API key in
//      the browser — proxy through chatApi to a server route.
//   3. Response: a tool_call -> AgentOutput with activeSchemaId = call name,
//      collected/arguments = JSON.parse(call.arguments), and toolCall set.
//      No tool_call yet -> return choice.content as the reply and keep the
//      prior activeSchemaId / collected / pendingField.

export type OpenAIFunctionTool = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, unknown>;
      required: string[];
    };
  };
};

function fieldToJsonSchema(def: FieldDef): Record<string, unknown> {
  if (def.type === "array") {
    return {
      type: "array",
      description: def.description,
      items: { type: def.items?.type === "integer" ? "integer" : "string" },
    };
  }
  const property: Record<string, unknown> = {
    type: def.type === "integer" ? "integer" : "string",
    description: def.description,
  };
  if (def.choices && def.choices.length > 0) {
    property.enum = def.choices;
  }
  return property;
}

/** Converts a Schema into an OpenAI function/tool definition. Your schema's
 * `parameters` map is already JSON-Schema-shaped, so this is a direct mapping. */
export function toOpenAITool(schema: Schema): OpenAIFunctionTool {
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  for (const [key, def] of Object.entries(schema.parameters)) {
    properties[key] = fieldToJsonSchema(def);
    required.push(key);
  }
  return {
    type: "function",
    function: {
      name: schema.id,
      description: schema.description,
      parameters: { type: "object", properties, required },
    },
  };
}
