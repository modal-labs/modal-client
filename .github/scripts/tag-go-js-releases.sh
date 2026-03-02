#!/usr/bin/env bash
set -euo pipefail


# Iterate through the commits
for commit in $COMMITS; do
    echo "Checking commit $commit"

  if git diff-tree --no-commit-id -r --name-only "$commit" | grep -q "^js/package.json$"; then
    VERSION=$(git show "$commit:js/package.json" | jq -r '.version')
    PREV_VERSION=$(git show "${commit}^:js/package.json" 2>/dev/null | jq -r '.version' 2>/dev/null || echo "")

    if [ "$VERSION" != "$PREV_VERSION" ]; then
      echo "Version changed to $VERSION at commit $commit, creating tags"
      JS_TAG="js/v$VERSION"
      if git rev-parse "$JS_TAG" >/dev/null 2>&1; then
        echo "Tags for $JS_TAG already exist, skipping"
      else
        echo "Tagging $JS_TAG"
        git tag "$JS_TAG" "$commit"
        git push origin "$JS_TAG"
      fi

      GO_TAG="go/v$VERSION"
      if git rev-parse "$GO_TAG" >/dev/null 2>&1; then
        echo "Tags for $GO_TAG already exist, skipping"
      else
        echo "Tagging $GO_TAG"
        git tag "$GO_TAG" "$commit"
        git push origin "$GO_TAG"
        curl "https://proxy.golang.org/github.com/modal-labs/modal-client/go/@v/v$VERSION.info"
      fi
    fi
  fi
done
