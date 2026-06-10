import { SignUp } from "@clerk/react";

function SignUpPage() {
  return (
    <div className="flex min-h-full flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-semibold sm:text-4xl">Create your account</h1>
      <SignUp routing="path" path="/sign-up" signInUrl="/sign-in" />
    </div>
  );
}

export default SignUpPage;
