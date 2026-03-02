#!/usr/bin/env bash
set -euo pipefail

# Iterate through the commits
for commit in $COMMITS; do
  echo "Checking commit $commit"
  # Check if py/modal_version/__init__.py was modified in this commit
  if git diff-tree --no-commit-id --name-only -r "$commit" | grep -q "^py/modal_version/__init__.py$"; then
    # Extract the version string from the file at this commit
    VERSION=$(git show "$commit:py/modal_version/__init__.py" | sed -n "s/^__version__ = [\"']\(.*\)[\"']/\1/p")
    PREV_VERSION=$(git show "${commit}^:py/modal_version/__init__.py" 2>/dev/null | sed -n "s/^__version__ = [\"']\(.*\)[\"']/\1/p" || echo "")

    if [ "$VERSION" != "$PREV_VERSION" ]; then
      PY_TAG="py/v$VERSION"
      if git rev-parse "$PY_TAG" >/dev/null 2>&1; then
        echo "Tags for $PY_TAG already exist, skipping"
      else
        echo "Tagging $PY_TAG"
        git tag "$PY_TAG" "$commit"
        git push origin "$PY_TAG"
      fi
    fi
  fi
done
