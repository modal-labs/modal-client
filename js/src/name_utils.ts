import { InvalidError } from "./errors";

const OBJECT_NAME_CHARSET = /^[a-zA-Z0-9\-_.]+$/;
const APP_ID_PATTERN = /^ap-[a-zA-Z0-9]{22}$/;

/**
 * Check whether a name is a valid Modal object name.
 *
 * @internal
 * @hidden
 */
export function isValidObjectName(name: string): boolean {
  return (
    name.length <= 64 &&
    OBJECT_NAME_CHARSET.test(name) &&
    !APP_ID_PATTERN.test(name)
  );
}

/**
 * Validate a Modal object name, throwing an {@link InvalidError} if invalid.
 * The `objectType` is used in the error message (e.g. "Image", "Image tag").
 *
 * @internal
 * @hidden
 */
export function checkObjectName(name: string, objectType: string): void {
  if (!isValidObjectName(name)) {
    throw new InvalidError(
      `Invalid ${objectType} name: '${name}'.` +
        "\n\nNames may contain only alphanumeric characters, dashes, periods, and underscores," +
        " must be shorter than 64 characters, and cannot conflict with App ID strings.",
    );
  }
}
