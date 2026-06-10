import { SignIn } from "@clerk/react";

function SignInPage() {
  return (
    <div className="flex min-h-full flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-semibold sm:text-4xl">Welcome back</h1>
      <SignIn routing="path" path="/sign-in" signUpUrl="/sign-up" />
    </div>
  );
}

export default SignInPage;
