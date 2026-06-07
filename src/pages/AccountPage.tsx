import { UserProfile } from "@clerk/react";

function AccountPage() {
  return (
    <div className="flex min-h-full items-center justify-center p-8">
      <UserProfile routing="path" path="/account" />
    </div>
  );
}

export default AccountPage;
