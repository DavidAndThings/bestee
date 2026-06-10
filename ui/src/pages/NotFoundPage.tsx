import { Link } from "react-router-dom";
import notFoundGif from "../assets/gifs/not-found.gif";

function NotFoundPage() {
  return (
    <div className="flex min-h-full flex-col items-center justify-center gap-4 p-8 text-center">
      <img
        src={notFoundGif}
        alt=""
        className="size-40 object-contain sm:size-48"
      />
      <p className="text-base-content/50 text-sm font-semibold tracking-widest uppercase">
        404
      </p>
      <h1 className="text-3xl font-bold sm:text-4xl">Page not found</h1>
      <p className="text-base-content/70 max-w-md">
        The page you were looking for doesn&rsquo;t exist or has moved.
      </p>
      <Link to="/" className="btn btn-primary mt-2">
        Back to Home
      </Link>
    </div>
  );
}

export default NotFoundPage;
