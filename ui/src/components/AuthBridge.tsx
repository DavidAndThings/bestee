import { useAuth } from "@clerk/react";
import { useEffect } from "react";
import { setTokenGetter } from "../services/auth";

/**
 * Registers Clerk's `getToken` with the API client's auth bridge so plain
 * (non-React) service modules can attach the session token to requests. Renders
 * nothing; mount once inside `ClerkProvider` (see `Layout`).
 */
export default function AuthBridge() {
  const { getToken } = useAuth();
  useEffect(() => {
    setTokenGetter(() => getToken());
    return () => setTokenGetter(null);
  }, [getToken]);
  return null;
}
