import { SignUp } from '@clerk/react'

function SignUpPage() {
  return (
    <div className="-mt-16 flex min-h-screen items-center justify-center p-8">
      <SignUp routing="path" path="/sign-up" signInUrl="/sign-in" />
    </div>
  )
}

export default SignUpPage
