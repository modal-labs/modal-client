## Describe your changes

- _Provide Linear issue reference (e.g. CLI-1234) if available._

<details> <summary>Checklists</summary>

---

## Compatibility checklist

Check these boxes or delete any item (or this section) if not relevant for this PR.

- [ ] Client+Server: this change is compatible with old servers
- [ ] Client forward compatibility: this change ensures client can accept data intended for later versions of itself

Note on protobuf: protobuf message changes in one place may have impact to
multiple entities (client, server, worker, database). See points above.


---

## Release checklist

If you intend for this commit to trigger a full release to PyPI, please ensure that they following steps have been taken:

- [ ] Version file has been updated with the next logical `X.Y.Z` version
- [ ] Changelog has been cleaned up and given an appropriate subhead

---

</details>

## Changelog

<!--
If relevant, include a brief user-facing description of what's new in this version.

Format the changelog updates using bullet points.
See https://modal.com/docs/reference/changelog for examples and try to use a consistent style.

Provide short code examples, indented under the relevant bullet point, if they would be helpful.
Cross-linking to relevant documentation is also encouraged.
-->
