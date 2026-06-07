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
    <div className="flex items-end gap-2">
      <textarea
        className="textarea textarea-bordered max-h-32 flex-1 resize-none"
        rows={1}
        value={text}
        placeholder={placeholder ?? "Type your answer…"}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
      />
      <button
        type="button"
        className="btn btn-primary"
        disabled={disabled || text.trim().length === 0}
        onClick={submit}
      >
        Send
      </button>
    </div>
  );
}

export default ChatInput;
