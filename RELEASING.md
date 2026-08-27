# Releasing the Modal SDKs

## JS and Go SDKs

1. Navigate to the client directory and run `inv update-version-go-js`:

```bash
cd client
# For major release
inv update-version-go-js --update major
# For minor release
inv update-version-go-js --update minor
# For patch release
inv update-version-go-js --update patch
```

2. Move `CHANGELOG_DEV.md` items into relevant language-specific changelogs, reordering and editing as needed.
3. Open PR with your changes.
