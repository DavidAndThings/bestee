import { SignUp } from "@clerk/react";

function SignUpPage() {
  return (
    <div className="flex min-h-full items-center justify-center p-8">
      <SignUp routing="path" path="/sign-up" signInUrl="/sign-in" />
    </div>
  );
}

export default SignUpPage;
