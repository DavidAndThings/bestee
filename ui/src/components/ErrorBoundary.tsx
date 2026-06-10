import { Component, type ErrorInfo, type ReactNode } from "react";
import { Link } from "react-router-dom";

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  hasError: boolean;
};

/**
 * Catches render-time errors in the route subtree so a broken page
 * doesn't blank out the entire app.  Lives inside the Layout outlet,
 * so the navbar stays visible and the user can navigate away.
 *
 * Has to be a class component — React's error boundary contract is
 * still class-only as of React 19.
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Surface to the browser console; if we ever wire up a remote
    // error reporter (Sentry, etc.), this is where it plugs in.
    console.error("Route render error:", error, errorInfo);
  }

  private handleRetry = () => {
    this.setState({ hasError: false });
  };

  render() {
    if (!this.state.hasError) return this.props.children;
    return (
      <div className="flex min-h-full flex-col items-center justify-center gap-4 p-8 text-center">
        <p className="text-error text-sm font-semibold tracking-widest uppercase">
          Error
        </p>
        <h1 className="text-3xl font-bold sm:text-4xl">Something went wrong</h1>
        <p className="text-base-content/70 max-w-md">
          The page hit an unexpected error.  Try again, or head back home.
        </p>
        <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
          <button
            type="button"
            onClick={this.handleRetry}
            className="btn btn-primary"
          >
            Try again
          </button>
          <Link to="/" className="btn" onClick={this.handleRetry}>
            Back to Home
          </Link>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
