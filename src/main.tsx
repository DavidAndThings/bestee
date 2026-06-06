import { ClerkProvider } from "@clerk/react";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import Layout from "./Layout.tsx";
import App from "./App.tsx";
import SignInPage from "./pages/SignInPage.tsx";
import SignUpPage from "./pages/SignUpPage.tsx";
import AccountPage from "./pages/AccountPage.tsx";

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

if (!PUBLISHABLE_KEY) {
  throw new Error("Missing VITE_CLERK_PUBLISHABLE_KEY in .env.local");
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ClerkProvider
      publishableKey={PUBLISHABLE_KEY}
      afterSignOutUrl="/"
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
      signInFallbackRedirectUrl="/"
      signUpFallbackRedirectUrl="/"
    >
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<App />} />
            <Route path="sign-in/*" element={<SignInPage />} />
            <Route path="sign-up/*" element={<SignUpPage />} />
            <Route path="account/*" element={<AccountPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ClerkProvider>
  </StrictMode>,
);
