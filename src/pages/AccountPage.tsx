import { UserProfile } from '@clerk/react'

function AccountPage() {
  return (
    <div className="-mt-16 flex min-h-screen items-center justify-center p-8">
      <UserProfile routing="path" path="/account" />
    </div>
  )
}

export default AccountPage
