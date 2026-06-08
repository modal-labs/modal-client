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

2. You can change the wording or order of items in `client/CHANGELOG_GO_JS.md`
3. Open PR with your changes.
