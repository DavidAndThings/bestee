import type { ChatMessage as ChatMessageType } from "../../lib/types";

export type ChatMessageProps = {
  message: ChatMessageType;
};

function ChatMessage({ message }: ChatMessageProps) {
  if (message.role === "system") {
    return (
      <div className="text-base-content/50 py-1 text-center text-xs">
        {message.content}
      </div>
    );
  }

  const isUser = message.role === "user";

  return (
    <div className={`chat ${isUser ? "chat-end" : "chat-start"}`}>
      <div
        className={`chat-bubble whitespace-pre-wrap ${
          isUser ? "chat-bubble-primary" : ""
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}

export default ChatMessage;
