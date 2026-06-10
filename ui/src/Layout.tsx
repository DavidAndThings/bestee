import { Show } from "@clerk/react";
import { Link, NavLink, Outlet } from "react-router-dom";
import UserMenu from "./components/UserMenu";
import openSign from "./assets/icons/open-sign.svg";
import homeIcon from "./assets/icons/home.svg";

function Layout() {
  return (
    <div className="flex h-dvh flex-col">
      <div className="navbar bg-base-100 shadow-sm px-0 shrink-0">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 px-8">
          <div
            className="flex-none text-4xl"
            style={{ fontFamily: "'Bitcount Grid Double Ink', system-ui" }}
          >
            Bestee
          </div>
          <div className="flex flex-none items-center gap-2">
            <ul className="menu menu-horizontal px-1">
              <li>
                <NavLink to="/" className="gap-2">
                  <img src={homeIcon} alt="" className="size-6" />
                  Home
                </NavLink>
              </li>
            </ul>
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
        <Outlet />
      </main>
    </div>
  );
}

export default Layout;
