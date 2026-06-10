import { UserProfile } from "@clerk/react";

function AccountPage() {
  return (
    <div className="flex min-h-full flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-semibold sm:text-4xl">Account settings</h1>
      <UserProfile routing="path" path="/account" />
    </div>
  );
}

export default AccountPage;
