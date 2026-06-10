import { RedirectToSignIn, useAuth } from "@clerk/react";
import { Outlet } from "react-router-dom";

/**
 * Route guard for signed-in routes. Waits for Clerk to load, then either renders the
 * nested routes (signed in) or redirects to sign-in (signed out). Clerk returns
 * the user to their original destination after authenticating.
 */
function RequireAuth() {
  const { isLoaded, isSignedIn } = useAuth();

  if (!isLoaded) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <span className="loading loading-spinner loading-lg" />
      </div>
    );
  }

  if (!isSignedIn) {
    return <RedirectToSignIn />;
  }

  return <Outlet />;
}

export default RequireAuth;
