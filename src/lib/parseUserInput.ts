import type { Question } from "./types";

export type ParseResult =
  | { ok: true; value: string | number }
  | { ok: false; error: string };

/**
 * Validates and coerces a single raw user answer for the given question.
 * For array questions this parses one item; the caller handles accumulation.
 */
export function parseScalar(question: Question, raw: string): ParseResult {
  const trimmed = raw.trim();

  if (!trimmed) {
    return { ok: false, error: "Please enter a value." };
  }

  if (question.valueType === "integer") {
    const value = Number(trimmed);
    if (!Number.isFinite(value) || !Number.isInteger(value)) {
      return {
        ok: false,
        error: `"${trimmed}" isn't a whole number — try something like 14.`,
      };
    }
    return { ok: true, value };
  }

  if (question.choices && question.choices.length > 0) {
    const match = question.choices.find(
      (choice) => choice.toLowerCase() === trimmed.toLowerCase(),
    );
    if (!match) {
      return {
        ok: false,
        error: `Please choose one of: ${question.choices.join(", ")}.`,
      };
    }
    return { ok: true, value: match };
  }

  return { ok: true, value: trimmed };
}

const DONE_WORDS = new Set(["done", "finish", "finished", "no more", "that's all"]);

/** Whether the user signalled they're finished adding items to an array. */
export function isDoneSignal(raw: string): boolean {
  return DONE_WORDS.has(raw.trim().toLowerCase());
}
