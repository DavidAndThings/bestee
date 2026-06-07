import { useState } from "react";

export type ChatInputProps = {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
};

function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [text, setText] = useState("");

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <textarea
      className="textarea textarea-bordered max-h-32 w-full resize-none"
      rows={1}
      value={text}
      placeholder={placeholder ?? "Type a message…"}
      disabled={disabled}
      onChange={(event) => setText(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          submit();
        }
      }}
    />
  );
}

export default ChatInput;
