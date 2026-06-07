import { useEffect, useRef } from "react";
import { getSchema, SCHEMA_REGISTRY } from "../../config/schemas";
import type { Conversation } from "../../lib/types";
import { schemaToQuestions } from "../../lib/schemaToQuestions";
import { useAgentChat } from "../../hooks/useAgentChat";
import ChatWindow from "./ChatWindow";
import ChatInput from "./ChatInput";
import QuickReplies, { type QuickReply } from "./QuickReplies";
import ConfirmationCard from "./ConfirmationCard";

export type ChatExperienceProps = {
  userId: string;
  initialConversation: Conversation | null;
  /** An opening message to auto-send once (e.g. from a home-page card). */
  seed?: string;
};

function ChatExperience({
  userId,
  initialConversation,
  seed,
}: ChatExperienceProps) {
  const chat = useAgentChat({ userId, initialConversation });
  const { conversation, isThinking, isSubmitting } = chat;

  const sentSeedRef = useRef(false);
  useEffect(() => {
    if (seed && !sentSeedRef.current) {
      sentSeedRef.current = true;
      chat.sendMessage(seed);
    }
  }, [seed, chat]);

  const activeSchema = getSchema(conversation.activeSchemaId ?? undefined);
  const activeQuestion =
    activeSchema && conversation.pendingField
      ? schemaToQuestions(activeSchema).find(
          (q) => q.fieldKey === conversation.pendingField,
        )
      : undefined;

  const replies: QuickReply[] = [];
  if (activeQuestion?.choices) {
    for (const choice of activeQuestion.choices) {
      replies.push({ label: choice, value: choice });
    }
  }
  if (activeQuestion?.isArray) {
    replies.push({ label: "✓ Done", value: "done" });
  }

  const placeholder = activeQuestion?.isArray
    ? 'Add a value, or say "done"'
    : activeQuestion?.choices
      ? "Pick an option or type it"
      : "Message the assistant…";

  const toolCall = conversation.pendingToolCall;
  const toolSchema = toolCall ? getSchema(toolCall.schemaId) : undefined;
  const busy = isThinking || isSubmitting;

  // When no tool is active, offer one-click prompts to start each tool.
  const toolSuggestions: QuickReply[] = activeSchema
    ? []
    : SCHEMA_REGISTRY.map((schema) => ({
        label: schema.name,
        value: `Let's set up a ${schema.name}.`,
      }));

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="border-base-300 border-b">
        <div className="px-4 py-3">
          <div className="mx-auto flex w-full max-w-2xl items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <h1 className="text-lg leading-tight font-semibold">Topic</h1>
              {activeSchema ? (
                <span className="badge badge-outline badge-sm">
                  {activeSchema.name}
                </span>
              ) : (
                <span className="badge badge-ghost badge-sm">No topic yet</span>
              )}
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={chat.reset}
              disabled={isSubmitting}
            >
              New chat
            </button>
          </div>
        </div>
      </header>

      <ChatWindow messages={conversation.messages} typing={busy} />

      <footer className="border-base-300 bg-base-100 max-h-[55%] overflow-y-auto border-t p-4">
        <div className="mx-auto w-full max-w-2xl">
          {toolCall && toolSchema ? (
            <ConfirmationCard
              schema={toolSchema}
              questions={schemaToQuestions(toolSchema)}
              values={toolCall.arguments}
              submitting={isSubmitting}
              onConfirm={chat.confirm}
              onEditField={chat.editField}
            />
          ) : (
            <div className="flex flex-col gap-3">
              {!activeSchema && (
                <div className="flex flex-col gap-1.5">
                  <span className="text-base-content/50 text-xs">
                    Quick start
                  </span>
                  <QuickReplies
                    replies={toolSuggestions}
                    onSelect={chat.sendMessage}
                    disabled={busy}
                  />
                </div>
              )}
              <QuickReplies
                replies={replies}
                onSelect={chat.sendMessage}
                disabled={busy}
              />
              <ChatInput
                onSend={chat.sendMessage}
                placeholder={placeholder}
                disabled={busy}
              />
            </div>
          )}
        </div>
      </footer>
    </div>
  );
}

export default ChatExperience;
