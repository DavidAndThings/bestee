import { useRef } from "react";
import { useClerk, useUser } from "@clerk/react";
import { useNavigate } from "react-router-dom";
import closeSign from "../assets/icons/close-sign.svg";
import testAccount from "../assets/icons/account.svg";

function UserMenu() {
  const { user } = useUser();
  const { signOut } = useClerk();
  const navigate = useNavigate();
  const dialogRef = useRef<HTMLDialogElement>(null);

  if (!user) return null;

  const closeMenu = () => {
    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  };

  const handleConfirm = () => {
    dialogRef.current?.close();
    void signOut({ redirectUrl: "/" });
  };

  return (
    <>
      <div className="dropdown dropdown-end">
        <div
          tabIndex={0}
          role="button"
          className="btn btn-ghost btn-circle avatar"
        >
          <div className="w-9 rounded-full">
            <img
              src={user.imageUrl}
              alt={user.fullName ?? user.primaryEmailAddress?.emailAddress ?? "User"}
            />
          </div>
        </div>
        <ul
          tabIndex={0}
          className="menu dropdown-content bg-base-100 rounded-box z-10 mt-3 w-52 p-2 shadow"
        >
          <li>
            <button
              type="button"
              onClick={() => {
                closeMenu();
                navigate("/account");
              }}
            >
              <img src={testAccount} alt="" className="size-6" />
              Manage account
            </button>
          </li>
          <li>
            <button
              type="button"
              onClick={() => {
                closeMenu();
                dialogRef.current?.showModal();
              }}
            >
              <img src={closeSign} alt="" className="size-6" />
              Sign out
            </button>
          </li>
        </ul>
      </div>

      <dialog ref={dialogRef} className="modal" aria-labelledby="signout-title">
        <div className="modal-box">
          <h3 id="signout-title" className="text-lg font-bold">Sign out?</h3>
          <p className="py-4">Are you sure you want to sign out?</p>
          <div className="modal-action">
            <form method="dialog">
              <button type="submit" className="btn">
                Cancel
              </button>
            </form>
            <button
              type="button"
              className="btn btn-error"
              onClick={handleConfirm}
            >
              Sign out
            </button>
          </div>
        </div>
        <form method="dialog" className="modal-backdrop">
          <button type="submit" aria-hidden="true">close</button>
        </form>
      </dialog>
    </>
  );
}

export default UserMenu;
