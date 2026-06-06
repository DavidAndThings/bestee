import { SignIn } from '@clerk/react'

function SignInPage() {
  return (
    <div className="-mt-16 flex min-h-screen items-center justify-center p-8">
      <SignIn routing="path" path="/sign-in" signUpUrl="/sign-up" />
    </div>
  )
}

export default SignInPage
