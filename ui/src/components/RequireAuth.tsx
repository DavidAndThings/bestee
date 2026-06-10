import { useAuth } from "@clerk/react";
import { Navigate, Outlet, useLocation } from "react-router-dom";

/**
 * Route guard for signed-in routes.  Waits for Clerk to load, then either
 * renders the nested routes (signed in) or redirects to sign-in (signed
 * out) with `redirect_url` set to the originally-requested path so the
 * user lands back where they intended after authenticating.
 */
function RequireAuth() {
  const { isLoaded, isSignedIn } = useAuth();
  const location = useLocation();

  if (!isLoaded) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <span className="loading loading-spinner loading-lg" />
      </div>
    );
  }

  if (!isSignedIn) {
    const target = `${location.pathname}${location.search}${location.hash}`;
    return (
      <Navigate
        to={`/sign-in?redirect_url=${encodeURIComponent(target)}`}
        replace
      />
    );
  }

  return <Outlet />;
}

export default RequireAuth;
