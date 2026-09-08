// This example configures a client using credentials from custom environment variables.

import { ModalClient, type ModalClientParams } from "modal";

const refreshToken = process.env.CUSTOM_MODAL_OAUTH_REFRESH_TOKEN;
let credentials: ModalClientParams;
if (refreshToken) {
  const oauthClientId = process.env.CUSTOM_MODAL_OAUTH_CLIENT_ID;
  if (!oauthClientId) {
    throw new Error(
      "CUSTOM_MODAL_OAUTH_CLIENT_ID environment variable not set",
    );
  }
  const oauthClientSecret = process.env.CUSTOM_MODAL_OAUTH_CLIENT_SECRET;
  if (!oauthClientSecret) {
    throw new Error(
      "CUSTOM_MODAL_OAUTH_CLIENT_SECRET environment variable not set",
    );
  }
  credentials = {
    oauthRefreshToken: refreshToken,
    oauthClientId,
    oauthClientSecret,
  };
} else {
  const modalId = process.env.CUSTOM_MODAL_ID;
  if (!modalId) {
    throw new Error("CUSTOM_MODAL_ID environment variable not set");
  }
  const modalSecret = process.env.CUSTOM_MODAL_SECRET;
  if (!modalSecret) {
    throw new Error("CUSTOM_MODAL_SECRET environment variable not set");
  }
  credentials = { tokenId: modalId, tokenSecret: modalSecret };
}

const modal = new ModalClient(credentials);

const echo = await modal.functions.fromName(
  "libmodal-test-support",
  "echo_string",
);
console.log(echo);
