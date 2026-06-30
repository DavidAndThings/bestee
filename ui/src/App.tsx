import { useEffect, useState } from "react";
import { useAuth } from "@clerk/react";
import { useNavigate } from "react-router-dom";
import Card from "./components/Card";
import { SCHEMA_REGISTRY } from "./config/schemas";
import pocketConsole from "./assets/gifs/pocket-console.gif";
import paperMoney from "./assets/gifs/paper-money.gif";

type HomeCard = {
  id: string;
  title: string;
  description: string;
  badge?: string;
  imageUrl?: string;
  actionLabel: string;
  onAction: () => void;
};

const FAVOURITES_KEY = "bestee:favourites";

/** Every card id, in declared order (schemas first, then Job Status). */
const ALL_CARD_IDS = [
  ...SCHEMA_REGISTRY.map((schema) => schema.id),
  "sic-codes",
  "job-status",
];

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

/** Card order for this visit: favourites first, declared order otherwise.
 * Captured once on load so toggling a heart doesn't reshuffle the grid mid-
 * session — the updated order only takes effect on the next load/refresh. */
function initialOrder(): string[] {
  const favs = loadFavourites();
  return [...ALL_CARD_IDS].sort((a, b) => {
    const aFav = favs.has(a) ? 1 : 0;
    const bFav = favs.has(b) ? 1 : 0;
    return bFav - aFav;
  });
}

function App() {
  const navigate = useNavigate();
  const { isSignedIn } = useAuth();
  const [favourites, setFavourites] = useState<Set<string>>(loadFavourites);
  // Frozen for the session; recomputed only on the next mount/refresh, so
  // favouriting reorders the grid on the *next* visit, not immediately.
  const [orderedIds] = useState<string[]>(initialOrder);

  // Persist favourites so the next visit can order by them.
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
      actionLabel: isSignedIn ? "Configure" : "Sign in to start",
      onAction: () => navigateWithAuth(`/charts/${schema.id}`),
    })),
    {
      id: "sic-codes",
      title: "SIC Code Directory",
      description:
        "Browse SIC industry codes and the tickers classified under each.",
      badge: "Reference",
      imageUrl: paperMoney,
      actionLabel: isSignedIn ? "Browse" : "Sign in to browse",
      onAction: () => navigateWithAuth("/sic"),
    },
    {
      id: "job-status",
      title: "Job Status",
      description: "Track the status of your submitted analysis jobs.",
      badge: "Jobs",
      imageUrl: pocketConsole,
      actionLabel: isSignedIn ? "View jobs" : "Sign in to view",
      onAction: () => navigateWithAuth("/jobs"),
    },
  ];

  // Render in the order captured at load (see initialOrder).
  const orderedCards = [...cards].sort(
    (a, b) => orderedIds.indexOf(a.id) - orderedIds.indexOf(b.id),
  );

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
