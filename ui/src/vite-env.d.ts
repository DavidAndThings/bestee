/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Clerk publishable key (required). */
  readonly VITE_CLERK_PUBLISHABLE_KEY: string;
  /**
   * Base URL of the bestee API (e.g. `https://app.example.com/api` or
   * `http://localhost:8000`). When unset, the UI uses the in-browser mock
   * backend instead of issuing real HTTP requests.
   */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
