import type { Schema } from "../config/schemas";

/**
 * Assembles the final JSON payload from the collected values, in the field
 * order declared by the schema. Only schema-defined fields are included.
 */
export function buildPayload(
  schema: Schema,
  collectedValues: Record<string, unknown>,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {};
  for (const key of Object.keys(schema.parameters)) {
    payload[key] = collectedValues[key];
  }
  return payload;
}
