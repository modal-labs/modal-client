# Test support for libmodal

Sign in to Modal, which you'll use for running the test programs.

Then deploy the apps and secrets in this folder using the Python SDK. This
requires being signed in to AWS (Modal Labs account):

```bash
test-support/setup.sh
```

Now you can run tests in each language.

```bash
# JavaScript
cd modal-js && npm run build && npm test

# Go
cd modal-go && go test -v -count=1 -parallel=10 . ./test
```
