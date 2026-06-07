import * as mockBackend from "./mockBackend";
import type { Conversation } from "../lib/types";

export type { SubmitInput, SubmitResult } from "./mockBackend";

/**
 * The application's data layer. Today it delegates to a localStorage-backed
 * mock, but this is the only module the UI talks to — point these functions at
 * a real API to go live without touching components or hooks.
 */
export const chatApi = {
  loadConversation(userId: string): Promise<Conversation | null> {
    return mockBackend.loadConversation(userId);
  },
  saveConversation(
    userId: string,
    conversation: Conversation,
  ): Promise<Conversation> {
    return mockBackend.saveConversation(userId, conversation);
  },
  submitPayload(
    userId: string,
    input: mockBackend.SubmitInput,
  ): Promise<mockBackend.SubmitResult> {
    return mockBackend.submitPayload(userId, input);
  },
};
