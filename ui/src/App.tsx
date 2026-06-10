import { useState } from "react";
import { useAuth } from "@clerk/react";
import { useNavigate } from "react-router-dom";
import Card from "./components/Card";
import { SCHEMA_REGISTRY } from "./config/schemas";
import pocketConsole from "./assets/gifs/pocket-console.gif";

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

  const navigateWithAuth = (target: string) => {
    if (isSignedIn) {
      navigate(target);
    } else {
      navigate(`/sign-in?redirect_url=${encodeURIComponent(target)}`);
    }
  };

  const openChartSetup = (schemaId: string) => {
    navigateWithAuth(`/charts/${schemaId}`);
  };

  const goToJobs = () => {
    navigateWithAuth("/jobs");
  };

  return (
    <div className="py-8">
      <div className="mx-auto grid w-full max-w-6xl grid-cols-1 gap-6 px-8 sm:grid-cols-2">
        {SCHEMA_REGISTRY.map((schema) => (
          <Card
            key={schema.id}
            title={schema.name}
            description={
              schema.badge
                ? `${schema.badge} · ${schema.description}`
                : schema.description
            }
            imageUrl={schema.imageUrl}
            buttonVariant={schema.buttonVariant}
            actionLabel={isSignedIn ? "Configure" : "Sign in to start"}
            onAction={() => openChartSetup(schema.id)}
            favourited={favourites.has(schema.id)}
            onFavourite={() => toggleFavourite(schema.id)}
          />
        ))}
        <Card
          key="job-status"
          title="Job Status"
          description="Track the status of your submitted analysis jobs."
          badge="Jobs"
          imageUrl={pocketConsole}
          buttonVariant="btn-info"
          actionLabel={isSignedIn ? "View jobs" : "Sign in to view"}
          onAction={goToJobs}
          favourited={favourites.has("job-status")}
          onFavourite={() => toggleFavourite("job-status")}
        />
      </div>
    </div>
  );
}

export default App;
