# Modal container dependencies

This directory contains requirements that enumerate the dependencies needed by
the Modal client library when it is running inside a Modal container.

The container requirements are a subset of the dependencies required by the
client for local operation (i.e., to run or deploy Modal apps). Additionally, we
pin specific versions rather than allowing a range as we do for the installation
dependencies.

From version `2024.04`, the requirements should pin the entire dependency tree
and not just the first-order dependencies.

Note that for `2023.12`, there is a separate requirements file that is used for
Python 3.12. Going forward, it would be preferable to have a single file per
image builder version, using Python version conditionals when necessary.