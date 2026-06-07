import { useCallback, useEffect, useReducer } from "react";
import { getSchema, SCHEMA_REGISTRY } from "../config/schemas";
import type {
  ChatMessage,
  ChatRole,
  Conversation,
} from "../lib/types";
import { schemaToQuestions } from "../lib/schemaToQuestions";
import { chatApi } from "../services/chatApi";
import { runAgentTurn, promptForField, type AgentOutput } from "../services/agent";

type EngineState = {
  conversation: Conversation;
  isThinking: boolean;
  isSubmitting: boolean;
};

type EngineAction =
  | { type: "APPEND_USER"; text: string }
  | { type: "THINKING"; value: boolean }
  | { type: "APPLY_AGENT"; output: AgentOutput }
  | { type: "AGENT_ERROR" }
  | { type: "SUBMIT_START" }
  | { type: "SUBMIT_SUCCESS"; requestId: string }
  | { type: "SUBMIT_ERROR" }
  | { type: "EDIT_FIELD"; field: string }
  | { type: "RESET"; conversation: Conversation };

function uid(): string {
  return crypto.randomUUID();
}

function makeMessage(role: ChatRole, content: string): ChatMessage {
  return { id: uid(), role, content, createdAt: Date.now() };
}

function withMessage(
  conversation: Conversation,
  message: ChatMessage,
): Conversation {
  return {
    ...conversation,
    messages: [...conversation.messages, message],
    updatedAt: Date.now(),
  };
}

function greeting(): string {
  const tools = SCHEMA_REGISTRY.map((schema) => schema.name).join(", ");
  return `Hi! I'm your analytics assistant. I can help you with: ${tools}. What would you like to do?`;
}

export function createFreshConversation(userId: string): Conversation {
  const now = Date.now();
  return {
    id: uid(),
    userId,
    messages: [makeMessage("assistant", greeting())],
    activeSchemaId: null,
    collected: {},
    pendingField: null,
    pendingToolCall: null,
    createdAt: now,
    updatedAt: now,
  };
}

function reducer(state: EngineState, action: EngineAction): EngineState {
  switch (action.type) {
    case "APPEND_USER":
      return {
        ...state,
        conversation: withMessage(
          state.conversation,
          makeMessage("user", action.text),
        ),
      };

    case "THINKING":
      return { ...state, isThinking: action.value };

    case "APPLY_AGENT": {
      const { output } = action;
      let conversation = state.conversation;
      if (output.reply) {
        conversation = withMessage(
          conversation,
          makeMessage("assistant", output.reply),
        );
      }
      conversation = {
        ...conversation,
        activeSchemaId: output.activeSchemaId,
        collected: output.collected,
        pendingField: output.pendingField,
        pendingToolCall: output.toolCall ?? null,
        updatedAt: Date.now(),
      };
      return { ...state, conversation, isThinking: false };
    }

    case "AGENT_ERROR":
      return {
        ...state,
        isThinking: false,
        conversation: withMessage(
          state.conversation,
          makeMessage(
            "assistant",
            "Sorry — something went wrong. Could you try that again?",
          ),
        ),
      };

    case "SUBMIT_START":
      return { ...state, isSubmitting: true };

    case "SUBMIT_SUCCESS": {
      const schemaId = state.conversation.pendingToolCall?.schemaId;
      const name = getSchema(schemaId ?? undefined)?.name ?? "request";
      const conversation: Conversation = {
        ...withMessage(
          state.conversation,
          makeMessage(
            "assistant",
            `Done! Your ${name} was submitted. Reference: ${action.requestId}. Anything else I can help with?`,
          ),
        ),
        activeSchemaId: null,
        collected: {},
        pendingField: null,
        pendingToolCall: null,
        updatedAt: Date.now(),
      };
      return { ...state, conversation, isSubmitting: false };
    }

    case "SUBMIT_ERROR":
      return {
        ...state,
        isSubmitting: false,
        conversation: withMessage(
          state.conversation,
          makeMessage(
            "assistant",
            "That didn't go through. Tap confirm to try again.",
          ),
        ),
      };

    case "EDIT_FIELD": {
      const { field } = action;
      const schema = getSchema(state.conversation.activeSchemaId ?? undefined);
      const question = schema
        ? schemaToQuestions(schema).find((q) => q.fieldKey === field)
        : undefined;
      const ask = question ? promptForField(question) : `What should ${field} be?`;
      const collected = { ...state.conversation.collected };
      delete collected[field];
      const conversation: Conversation = {
        ...withMessage(
          state.conversation,
          makeMessage("assistant", `Sure — let's update that. ${ask}`),
        ),
        collected,
        pendingField: field,
        pendingToolCall: null,
        updatedAt: Date.now(),
      };
      return { ...state, conversation };
    }

    case "RESET":
      return {
        conversation: action.conversation,
        isThinking: false,
        isSubmitting: false,
      };

    default:
      return state;
  }
}

export type UseAgentChatArgs = {
  userId: string;
  initialConversation: Conversation | null;
};

export type AgentChat = {
  conversation: Conversation;
  isThinking: boolean;
  isSubmitting: boolean;
  sendMessage: (text: string) => void;
  confirm: () => void;
  editField: (field: string) => void;
  reset: () => void;
};

export function useAgentChat({
  userId,
  initialConversation,
}: UseAgentChatArgs): AgentChat {
  const [state, dispatch] = useReducer(reducer, null, () => ({
    conversation: initialConversation ?? createFreshConversation(userId),
    isThinking: false,
    isSubmitting: false,
  }));

  const conversation = state.conversation;

  useEffect(() => {
    void chatApi.saveConversation(userId, conversation);
  }, [conversation, userId]);

  const sendMessage = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || state.isThinking || state.isSubmitting) return;

      const conv = state.conversation;
      dispatch({ type: "APPEND_USER", text: trimmed });
      dispatch({ type: "THINKING", value: true });
      try {
        const output = await runAgentTurn({
          schemas: SCHEMA_REGISTRY,
          activeSchemaId: conv.activeSchemaId,
          collected: conv.collected,
          pendingField: conv.pendingField,
          history: conv.messages,
          userMessage: trimmed,
        });
        dispatch({ type: "APPLY_AGENT", output });
      } catch {
        dispatch({ type: "AGENT_ERROR" });
      }
    },
    [state.conversation, state.isThinking, state.isSubmitting],
  );

  const confirm = useCallback(async () => {
    const conv = state.conversation;
    if (!conv.pendingToolCall || state.isSubmitting) return;

    const toolCall = conv.pendingToolCall;
    dispatch({ type: "SUBMIT_START" });
    try {
      const result = await chatApi.submitPayload(userId, {
        schemaId: toolCall.schemaId,
        payload: toolCall.arguments,
      });
      dispatch({ type: "SUBMIT_SUCCESS", requestId: result.requestId });
    } catch {
      dispatch({ type: "SUBMIT_ERROR" });
    }
  }, [state.conversation, state.isSubmitting, userId]);

  const editField = useCallback((field: string) => {
    dispatch({ type: "EDIT_FIELD", field });
  }, []);

  const reset = useCallback(() => {
    dispatch({ type: "RESET", conversation: createFreshConversation(userId) });
  }, [userId]);

  return {
    conversation: state.conversation,
    isThinking: state.isThinking,
    isSubmitting: state.isSubmitting,
    sendMessage,
    confirm,
    editField,
    reset,
  };
}
