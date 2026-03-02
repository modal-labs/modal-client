import { tc } from "../test-support/test-client";
import { expect, test, onTestFinished } from "vitest";

test("WriteAndReadBinaryFile", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const writeHandle = await sb.open("/tmp/test.bin", "w");
  await writeHandle.write(testData);
  await writeHandle.close();

  const readHandle = await sb.open("/tmp/test.bin", "r");
  const readData = await readHandle.read();
  expect(readData).toEqual(testData);
  await readHandle.close();
});

test("AppendToFileBinary", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const testData = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  const writeHandle = await sb.open("/tmp/append.txt", "w");
  await writeHandle.write(testData);
  await writeHandle.close();

  const moreTestData = new Uint8Array([7, 8, 9, 10]);
  const appendHandle = await sb.open("/tmp/append.txt", "a");
  await appendHandle.write(moreTestData);
  await appendHandle.close();

  const readHandle = await sb.open("/tmp/append.txt", "r");
  const content = await readHandle.read();
  const expectedData = new Uint8Array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 8, 9, 10,
  ]);
  expect(content).toEqual(expectedData);
  await readHandle.close();
});

test("FileHandleFlush", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const encodedData = new TextEncoder().encode("Test data");

  const handle = await sb.open("/tmp/flush.txt", "w");
  await handle.write(encodedData);
  await handle.flush();
  await handle.close();

  const readHandle = await sb.open("/tmp/flush.txt", "r");
  const content = await readHandle.read();
  expect(content).toEqual(encodedData);
  await readHandle.close();
});

test("MultipleFileOperations", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const encoder = new TextEncoder();
  const content1 = encoder.encode("File 1 content");
  const handle1 = await sb.open("/tmp/file1.txt", "w");
  await handle1.write(content1);
  await handle1.close();

  const handle2 = await sb.open("/tmp/file2.txt", "w");
  const content2 = encoder.encode("File 2 content");
  await handle2.write(content2);
  await handle2.close();

  const read1 = await sb.open("/tmp/file1.txt", "r");
  const readContent1 = await read1.read();
  await read1.close();

  const read2 = await sb.open("/tmp/file2.txt", "r");
  const readContent2 = await read2.read();
  await read2.close();

  expect(readContent1).toEqual(content1);
  expect(readContent2).toEqual(content2);
});

test("FileOpenModes", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const encoder = new TextEncoder();
  const content1 = encoder.encode("Initial content");
  const writeHandle = await sb.open("/tmp/modes.txt", "w");
  await writeHandle.write(content1);
  await writeHandle.close();

  const readHandle = await sb.open("/tmp/modes.txt", "r");
  const readContent1 = await readHandle.read();
  expect(readContent1).toEqual(content1);
  await readHandle.close();

  const appendContent = encoder.encode(" appended");
  const appendHandle = await sb.open("/tmp/modes.txt", "a");
  await appendHandle.write(appendContent);
  await appendHandle.close();

  const expectedContent = encoder.encode("Initial content appended");
  const finalRead = await sb.open("/tmp/modes.txt", "r");
  const finalContent = await finalRead.read();
  expect(finalContent).toEqual(expectedContent);
  await finalRead.close();
});

test("LargeFileOperations", async () => {
  const app = await tc.apps.fromName("libmodal-test", {
    createIfMissing: true,
  });
  const image = await tc.images.fromRegistry("alpine:3.21").build(app);
  const sb = await tc.sandboxes.create(app, image);
  onTestFinished(async () => await sb.terminate());

  const encoder = new TextEncoder();
  const largeData = encoder.encode("x".repeat(1000));

  const writeHandle = await sb.open("/tmp/large.txt", "w");
  await writeHandle.write(largeData);
  await writeHandle.close();

  const readHandle = await sb.open("/tmp/large.txt", "r");
  const content = await readHandle.read();
  expect(content).toEqual(largeData);
  expect(content.length).toBe(1000);
  await readHandle.close();
});
