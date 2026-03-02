export function checkForRenamedParams(
  params: any,
  renames: Record<string, string>,
): void {
  if (!params) return;

  for (const [oldName, newName] of Object.entries(renames)) {
    if (oldName in params) {
      throw new Error(
        `Parameter '${oldName}' has been renamed to '${newName}'.`,
      );
    }
  }
}
