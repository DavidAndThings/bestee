import { useState } from "react";
import { useAuth } from "@clerk/react";
import { useNavigate } from "react-router-dom";
import Card from "./components/Card";
import { SCHEMA_REGISTRY } from "./config/schemas";

function App() {
  const navigate = useNavigate();
  const { isSignedIn } = useAuth();
  const [favourites, setFavourites] = useState<Set<string>>(new Set());

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

  const startChat = (schemaName: string) => {
    if (isSignedIn) {
      navigate("/chat", { state: { seed: `Let's set up a ${schemaName}.` } });
    } else {
      navigate("/sign-in");
    }
  };

  return (
    <div className="p-8">
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-6 sm:grid-cols-2">
        {SCHEMA_REGISTRY.map((schema) => (
          <Card
            key={schema.id}
            title={schema.name}
            description={schema.description}
            badge={schema.badge}
            imageUrl={schema.imageUrl}
            buttonVariant={schema.buttonVariant}
            actionLabel={isSignedIn ? "Start chat" : "Sign in to start"}
            onAction={() => startChat(schema.name)}
            favourited={favourites.has(schema.id)}
            onFavourite={() => toggleFavourite(schema.id)}
          />
        ))}
      </div>
    </div>
  );
}

export default App;
