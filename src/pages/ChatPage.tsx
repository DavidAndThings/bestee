import { useEffect, useState } from "react";
import { useAuth } from "@clerk/react";
import { useLocation } from "react-router-dom";
import type { Conversation } from "../lib/types";
import { chatApi } from "../services/chatApi";
import ChatExperience from "../components/chat/ChatExperience";

type LocationState = { seed?: string } | null;

function ChatPage() {
  const { userId } = useAuth();
  const location = useLocation();

  // Capture an opening message handed over by a home-page card, once.
  const [seed] = useState<string | undefined>(
    () => (location.state as LocationState)?.seed,
  );

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!userId) return;
    let cancelled = false;
    void chatApi.loadConversation(userId).then((loaded) => {
      if (cancelled) return;
      setConversation(loaded);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [userId]);

  if (!userId || loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <span className="loading loading-spinner loading-lg" />
      </div>
    );
  }

  return (
    <div className="h-full">
      <ChatExperience
        key={userId}
        userId={userId}
        initialConversation={conversation}
        seed={seed}
      />
    </div>
  );
}

export default ChatPage;
