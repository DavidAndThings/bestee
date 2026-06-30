import { badgeClass } from "../lib/badge";

export type CardProps = {
  title: string;
  description: string;
  badge?: string;
  imageUrl?: string;
  actionLabel?: string;
  buttonVariant?:
    | "btn-primary"
    | "btn-secondary"
    | "btn-accent"
    | "btn-info"
    | "btn-success"
    | "btn-warning"
    | "btn-error";
  onAction?: () => void;
  onFavourite?: () => void;
  favourited?: boolean;
};

function Card({
  title,
  description,
  badge,
  imageUrl,
  actionLabel = "Start",
  buttonVariant = "btn-primary",
  onAction,
  onFavourite,
  favourited = false,
}: CardProps) {
  const favouriteLabel = favourited
    ? "Remove from favourites"
    : "Add to favourites";
  return (
    <div className="card bg-base-100 min-h-72 shadow-md transition-shadow hover:shadow-xl">
      <figure className="bg-base-200 relative h-40 overflow-hidden">
        {imageUrl ? (
          <img
            src={imageUrl}
            alt={title}
            loading="lazy"
            decoding="async"
            className="h-full w-full object-contain p-4"
          />
        ) : (
          <div className="from-primary/30 to-secondary/30 flex h-full w-full items-center justify-center bg-linear-to-br">
            <span className="text-base-content/40 text-sm">Image</span>
          </div>
        )}
        <div
          className="tooltip tooltip-right absolute top-3 left-3"
          data-tip={favouriteLabel}
        >
          <button
            type="button"
            className={`btn btn-circle btn-sm bg-base-100/80 hover:bg-base-100 border-0 shadow-sm backdrop-blur-sm${
              favourited ? " text-error" : ""
            }`}
            onClick={() => onFavourite?.()}
            aria-label={favouriteLabel}
            aria-pressed={favourited}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill={favourited ? "currentColor" : "none"}
              stroke="currentColor"
              strokeWidth={1.5}
              className="size-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12Z"
              />
            </svg>
          </button>
        </div>
        {badge && (
          <span
            className={`badge ${badgeClass(badge)} absolute top-3 right-3 shadow-sm`}
          >
            {badge}
          </span>
        )}
      </figure>
      <div className="card-body">
        <h2 className="card-title text-2xl">{title}</h2>
        <p className="text-base-content/70">{description}</p>
        <div className="card-actions mt-2 justify-end">
          <button
            type="button"
            className={`btn ${buttonVariant}`}
            onClick={() => onAction?.()}
          >
            {actionLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Card;
