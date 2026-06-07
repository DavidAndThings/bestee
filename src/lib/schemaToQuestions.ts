import type { Schema } from "../config/schemas";
import type { Question, ScalarType } from "./types";

/**
 * Converts a schema's `parameters` map into an ordered list of questions the
 * chat engine walks through. Field order follows the object's key order.
 */
export function schemaToQuestions(schema: Schema): Question[] {
  return Object.entries(schema.parameters).map(([fieldKey, def]) => {
    const isArray = def.type === "array";
    const rawType = isArray ? def.items?.type : def.type;
    const valueType: ScalarType = rawType === "integer" ? "integer" : "string";

    return {
      fieldKey,
      prompt: def.description,
      valueType,
      isArray,
      choices: def.choices,
    };
  });
}
