import type { Conversation } from "../lib/types";

/**
 * A stand-in for a real backend. The user's single conversation is persisted to
 * localStorage, namespaced per Clerk user id, and every operation simulates
 * network latency. Swap this file's internals for real HTTP calls without
 * touching the rest of the app (see `chatApi.ts`, the single integration point).
 */

const STORAGE_PREFIX = "bestee:chat:";

function delay<T>(value: T, ms = 250): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), ms));
}

function storageKey(userId: string): string {
  return `${STORAGE_PREFIX}${userId}`;
}

export function loadConversation(userId: string): Promise<Conversation | null> {
  try {
    const raw = localStorage.getItem(storageKey(userId));
    if (!raw) return delay(null);
    const parsed = JSON.parse(raw) as unknown;
    const valid =
      parsed &&
      typeof parsed === "object" &&
      Array.isArray((parsed as Conversation).messages);
    return delay(valid ? (parsed as Conversation) : null);
  } catch {
    return delay(null);
  }
}

export function saveConversation(
  userId: string,
  conversation: Conversation,
): Promise<Conversation> {
  localStorage.setItem(storageKey(userId), JSON.stringify(conversation));
  return delay(conversation, 120);
}

export type SubmitInput = {
  schemaId: string;
  payload: Record<string, unknown>;
};

export type SubmitResult = {
  ok: boolean;
  requestId: string;
  receivedAt: number;
};

/** Pretends to dispatch a completed tool call's payload to a backend. */
export function submitPayload(
  userId: string,
  input: SubmitInput,
): Promise<SubmitResult> {
  // In a real backend this would POST to an endpoint. Here we just log it.
  console.info("[mockBackend] submit", {
    userId,
    schemaId: input.schemaId,
    payload: input.payload,
  });
  return delay(
    {
      ok: true,
      requestId: crypto.randomUUID(),
      receivedAt: Date.now(),
    },
    900,
  );
}
