import { useState } from "react";
import Card, { type CardProps } from "./components/Card";
import paperMoney from "./assets/gifs/paper-money.gif";
import coins from "./assets/gifs/coins.gif";
import pencil from "./assets/gifs/pencil.gif";
import barChart from "./assets/gifs/bar-chart.gif";

const cards: Omit<CardProps, "favourited" | "onFavourite">[] = [
  {
    title: "Morning Routine",
    description:
      "Kick off your day with a curated set of habits and reminders tailored to you.",
    badge: "Daily",
    imageUrl: paperMoney,
    buttonVariant: "btn-primary",
  },
  {
    title: "Focus Sessions",
    description:
      "Block out distractions and dive deep with guided focus timers and ambient sounds.",
    badge: "Productivity",
    imageUrl: coins,
    buttonVariant: "btn-secondary",
  },
  {
    title: "Meal Planner",
    description:
      "Plan balanced meals for the week and generate a smart shopping list in seconds.",
    badge: "Wellness",
    imageUrl: pencil,
    buttonVariant: "btn-accent",
  },
  {
    title: "Weekly Review",
    description:
      "Reflect on your progress, celebrate wins, and set intentions for the week ahead.",
    badge: "Reflection",
    imageUrl: barChart,
    buttonVariant: "btn-success",
  },
];

function App() {
  const [favourites, setFavourites] = useState<Set<string>>(new Set());

  const toggleFavourite = (title: string) =>
    setFavourites((prev) => {
      const next = new Set(prev);
      if (next.has(title)) {
        next.delete(title);
      } else {
        next.add(title);
      }
      return next;
    });

  return (
    <div className="p-8">
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 sm:grid-cols-2">
        {cards.map((card) => (
          <Card
            key={card.title}
            {...card}
            favourited={favourites.has(card.title)}
            onFavourite={() => toggleFavourite(card.title)}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
