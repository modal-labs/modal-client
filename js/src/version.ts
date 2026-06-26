/**
 * The checked-in version of the Modal JS SDK.
 *
 * Keep this in sync with the `version` field in package.json, which is the
 * source of truth the release tooling tags both the JS and Go SDKs from. The
 * `inv lint-versions` linter enforces that they match.
 */
const SDK_VERSION = "0.8.1";

export function getSDKVersion(): string {
  return SDK_VERSION;
}
