/**
 * Bridges Clerk's session token into the plain-module API client.
 *
 * `jobsApi` / `httpBackend` are plain modules (not React components), so they
 * can't call `useAuth()`. A small `<AuthBridge>` component registers Clerk's
 * `getToken` here once, and the HTTP client reads it via `getAuthToken()` to
 * attach `Authorization: Bearer <token>` without threading hooks everywhere.
 */

type TokenGetter = () => Promise<string | null>;

let tokenGetter: TokenGetter | null = null;

/** Register (or clear, with `null`) the function that returns a fresh token. */
export function setTokenGetter(getter: TokenGetter | null): void {
  tokenGetter = getter;
}

/** The current Clerk session token, or `null` when signed out / unregistered. */
export async function getAuthToken(): Promise<string | null> {
  return tokenGetter ? tokenGetter() : null;
}
