export type QuickReply = {
  label: string;
  value: string;
};

export type QuickRepliesProps = {
  replies: QuickReply[];
  onSelect: (value: string) => void;
  disabled?: boolean;
};

function QuickReplies({ replies, onSelect, disabled }: QuickRepliesProps) {
  if (replies.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {replies.map((reply) => (
        <button
          key={reply.value}
          type="button"
          className="btn btn-sm btn-outline"
          disabled={disabled}
          onClick={() => onSelect(reply.value)}
        >
          {reply.label}
        </button>
      ))}
    </div>
  );
}

export default QuickReplies;
