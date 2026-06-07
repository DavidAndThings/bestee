import { useEffect, useRef, useState } from "react";

export type ChatInputProps = {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
};

function ChatInput({ onSend, disabled, placeholder }: ChatInputProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Focus the input whenever it's ready (on mount and after each agent turn),
  // so it's immediately typable on landing here from a card.
  useEffect(() => {
    if (!disabled) {
      textareaRef.current?.focus();
    }
  }, [disabled]);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <textarea
      ref={textareaRef}
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
