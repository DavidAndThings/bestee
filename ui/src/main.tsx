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
import ChartSetupPage from "./pages/ChartSetupPage.tsx";
import JobsPage from "./pages/JobsPage.tsx";
import ResultTablePage from "./pages/ResultTablePage.tsx";
import SicCodesPage from "./pages/SicCodesPage.tsx";
import NotFoundPage from "./pages/NotFoundPage.tsx";
import RequireAuth from "./components/RequireAuth.tsx";

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
            <Route element={<RequireAuth />}>
              <Route path="charts/:schemaId" element={<ChartSetupPage />} />
              <Route path="jobs" element={<JobsPage />} />
              <Route
                path="results/:resultId/:aspect"
                element={<ResultTablePage />}
              />
              <Route path="sic" element={<SicCodesPage />} />
            </Route>
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ClerkProvider>
  </StrictMode>,
);
