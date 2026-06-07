import { useEffect, useRef } from "react";
import type { ChatMessage as ChatMessageType } from "../../lib/types";
import ChatMessage from "./ChatMessage";
import TypingIndicator from "./TypingIndicator";

export type ChatWindowProps = {
  messages: ChatMessageType[];
  typing?: boolean;
};

function ChatWindow({ messages, typing }: ChatWindowProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto flex w-full max-w-2xl flex-col gap-1">
        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}
        {typing && <TypingIndicator />}
        <div ref={endRef} />
      </div>
    </div>
  );
}

export default ChatWindow;
