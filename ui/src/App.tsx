import { useEffect, useState } from "react";
import { useAuth } from "@clerk/react";
import { useNavigate } from "react-router-dom";
import Card from "./components/Card";
import { SCHEMA_REGISTRY, type ButtonVariant } from "./config/schemas";
import pocketConsole from "./assets/gifs/pocket-console.gif";

type HomeCard = {
  id: string;
  title: string;
  description: string;
  badge?: string;
  imageUrl?: string;
  buttonVariant?: ButtonVariant;
  actionLabel: string;
  onAction: () => void;
};

const FAVOURITES_KEY = "bestee:favourites";

/** Read the persisted favourite ids (browser-local), defaulting to none. */
function loadFavourites(): Set<string> {
  try {
    const raw = localStorage.getItem(FAVOURITES_KEY);
    const ids: unknown = raw ? JSON.parse(raw) : [];
    if (!Array.isArray(ids)) return new Set();
    return new Set(ids.filter((id): id is string => typeof id === "string"));
  } catch {
    return new Set();
  }
}

function App() {
  const navigate = useNavigate();
  const { isSignedIn } = useAuth();
  const [favourites, setFavourites] = useState<Set<string>>(loadFavourites);

  // Persist favourites so they survive reloads.
  useEffect(() => {
    try {
      localStorage.setItem(FAVOURITES_KEY, JSON.stringify([...favourites]));
    } catch {
      // Storage may be unavailable (private mode, quota) — ignore.
    }
  }, [favourites]);

  const toggleFavourite = (id: string) =>
    setFavourites((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });

  const navigateWithAuth = (target: string) => {
    if (isSignedIn) {
      navigate(target);
    } else {
      navigate(`/sign-in?redirect_url=${encodeURIComponent(target)}`);
    }
  };

  const cards: HomeCard[] = [
    ...SCHEMA_REGISTRY.map((schema) => ({
      id: schema.id,
      title: schema.name,
      description: schema.description,
      badge: schema.badge,
      imageUrl: schema.imageUrl,
      buttonVariant: schema.buttonVariant,
      actionLabel: isSignedIn ? "Configure" : "Sign in to start",
      onAction: () => navigateWithAuth(`/charts/${schema.id}`),
    })),
    {
      id: "job-status",
      title: "Job Status",
      description: "Track the status of your submitted analysis jobs.",
      badge: "Jobs",
      imageUrl: pocketConsole,
      buttonVariant: "btn-info",
      actionLabel: isSignedIn ? "View jobs" : "Sign in to view",
      onAction: () => navigateWithAuth("/jobs"),
    },
  ];

  // Favourites first. Original order is preserved within each group because
  // Array.prototype.sort is stable.
  const orderedCards = [...cards].sort((a, b) => {
    const aFav = favourites.has(a.id) ? 1 : 0;
    const bFav = favourites.has(b.id) ? 1 : 0;
    return bFav - aFav;
  });

  return (
    <div className="py-8">
      <div className="mx-auto grid w-full max-w-7xl grid-cols-1 gap-8 px-8 sm:grid-cols-2 lg:grid-cols-3">
        {orderedCards.map((card) => (
          <Card
            key={card.id}
            title={card.title}
            description={card.description}
            badge={card.badge}
            imageUrl={card.imageUrl}
            buttonVariant={card.buttonVariant}
            actionLabel={card.actionLabel}
            onAction={card.onAction}
            favourited={favourites.has(card.id)}
            onFavourite={() => toggleFavourite(card.id)}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
