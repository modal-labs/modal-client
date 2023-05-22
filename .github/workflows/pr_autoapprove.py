# Copyright Modal Labs 2023
"""
Auto-approve hotfix pull requests.

SOC-2 requires that all Github pull requests have 1 independent approval.

> SOC 2 CM-03
> System changes are approved by at least 1 independent person prior to deployment into production.

We don't want hotfixes blocked on approval because they're 'emergency' changes,
so we use this script and a Github Actions workflow (.github/workflows/pr-autoapprove.yml)
to ensure hotfixes are immediately approved.
"""
import argparse
import os
import sys

import httpx

owner = "modal-labs"
repo = "modal-client"


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("title", type=str)
    parser.add_argument("number", type=str)
    args = parser.parse_args(argv)

    title = args.title.lower()
    pull_number = args.number

    if "hotfix" not in title:
        print(f"Ignoring PR {pull_number} with title '{title}'. Does not include 'HOTFIX' string in title.")
        return

    # GITHUB_TOKEN should be set in the environment of the Github
    # actions workflow step.
    token = os.getenv("GITHUB_TOKEN")

    # URL for the "submit a review" endpoint
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/reviews"

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    data = {
        "event": "APPROVE",
        "body": "Approved :+1:",
    }

    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    reviews = response.json()
    approvals = [r for r in reviews if r["state"] == "APPROVED"]
    if len(approvals) > 0:
        print(f"Ignoring PR {pull_number} because it has at least one approval already.")
        return

    response = httpx.post(url, headers=headers, json=data)

    if response.status_code == 200:
        print("Pull request approved successfully")
    else:
        print("Failed to approve pull request", response.content)


if __name__ == "__main__":
    main(sys.argv[1:])
