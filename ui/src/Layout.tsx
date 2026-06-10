import { Show } from "@clerk/react";
import { Link, Outlet, useLocation } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import UserMenu from "./components/UserMenu";
import openSign from "./assets/icons/open-sign.svg";

function Layout() {
  // Re-mount the ErrorBoundary on navigation so a captured error
  // doesn't stay sticky when the user moves to a different route.
  const location = useLocation();
  return (
    <div className="flex h-dvh flex-col">
      <div className="navbar bg-base-100 shadow-sm px-0 shrink-0">
        <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center justify-between gap-4 px-8">
          <Link
            to="/"
            aria-label="Home"
            className="flex-none text-4xl transition-opacity hover:opacity-70"
            style={{ fontFamily: "'Bitcount Grid Double Ink', system-ui" }}
          >
            Bestee
          </Link>
          <div className="flex flex-none items-center gap-2">
            <Show when="signed-out">
              <Link to="/sign-in" className="btn btn-ghost btn-sm gap-2">
                <img src={openSign} alt="" className="size-6" />
                Sign in
              </Link>
            </Show>
            <Show when="signed-in">
              <UserMenu />
            </Show>
          </div>
        </div>
      </div>

      <main className="min-h-0 flex-1 overflow-y-auto">
        <ErrorBoundary key={location.pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default Layout;
