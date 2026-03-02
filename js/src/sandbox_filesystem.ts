import {
  ContainerFilesystemExecRequest,
  DeepPartial,
  ContainerFilesystemExecResponse,
} from "../proto/modal_proto/api";
import { type ModalClient, ModalGrpcClient, isRetryableGrpc } from "./client";
import { SandboxFilesystemError } from "./errors";

/** File open modes supported by the filesystem API. */
export type SandboxFileMode = "r" | "w" | "a" | "r+" | "w+" | "a+";

/**
 * SandboxFile represents an open file in the {@link Sandbox} filesystem.
 * Provides read/write operations similar to Node.js `fsPromises.FileHandle`.
 */
export class SandboxFile {
  readonly #client: ModalClient;
  readonly #fileDescriptor: string;
  readonly #taskId: string;

  /** @ignore */
  constructor(client: ModalClient, fileDescriptor: string, taskId: string) {
    this.#client = client;
    this.#fileDescriptor = fileDescriptor;
    this.#taskId = taskId;
  }

  /**
   * Read data from the file.
   * @returns Promise that resolves to the read data as Uint8Array
   */
  async read(): Promise<Uint8Array> {
    const resp = await runFilesystemExec(this.#client.cpClient, {
      fileReadRequest: {
        fileDescriptor: this.#fileDescriptor,
      },
      taskId: this.#taskId,
    });
    const chunks = resp.chunks;

    // Concatenate all chunks into a single Uint8Array
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.length;
    }

    return result;
  }

  /**
   * Write data to the file.
   * @param data - Data to write (string or Uint8Array)
   */
  async write(data: Uint8Array): Promise<void> {
    await runFilesystemExec(this.#client.cpClient, {
      fileWriteRequest: {
        fileDescriptor: this.#fileDescriptor,
        data,
      },
      taskId: this.#taskId,
    });
  }

  /**
   * Flush any buffered data to the file.
   */
  async flush(): Promise<void> {
    await runFilesystemExec(this.#client.cpClient, {
      fileFlushRequest: {
        fileDescriptor: this.#fileDescriptor,
      },
      taskId: this.#taskId,
    });
  }

  /**
   * Close the file handle.
   */
  async close(): Promise<void> {
    await runFilesystemExec(this.#client.cpClient, {
      fileCloseRequest: {
        fileDescriptor: this.#fileDescriptor,
      },
      taskId: this.#taskId,
    });
  }
}

export async function runFilesystemExec(
  cpClient: ModalGrpcClient,
  request: DeepPartial<ContainerFilesystemExecRequest>,
): Promise<{
  chunks: Uint8Array[];
  response: ContainerFilesystemExecResponse;
}> {
  const response = await cpClient.containerFilesystemExec(request);

  const chunks: Uint8Array[] = [];
  let retries = 10;
  let completed = false;
  while (!completed) {
    try {
      const outputIterator = cpClient.containerFilesystemExecGetOutput({
        execId: response.execId,
        timeout: 55,
      });
      for await (const batch of outputIterator) {
        chunks.push(...batch.output);
        if (batch.eof) {
          completed = true;
          break;
        }
        if (batch.error !== undefined) {
          if (retries > 0) {
            retries--;
            break;
          }
          throw new SandboxFilesystemError(batch.error.errorMessage);
        }
      }
    } catch (err) {
      if (isRetryableGrpc(err) && retries > 0) {
        retries--;
      } else throw err;
    }
  }
  return { chunks, response };
}
