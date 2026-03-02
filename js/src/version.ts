declare const __MODAL_SDK_VERSION__: string;

export function getSDKVersion(): string {
  return typeof __MODAL_SDK_VERSION__ !== "undefined"
    ? __MODAL_SDK_VERSION__
    : "0.0.0";
}
