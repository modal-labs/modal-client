import { expect, test } from "vitest";
import { Retries } from "modal";
import { parseRetries } from "../../src/retries";

test("parseRetries", async () => {
  const r = parseRetries(3)!;
  expect(r).toBeDefined();
  expect(r.maxRetries).toBe(3);
  expect(r.backoffCoefficient).toBe(1.0);
  expect(r.initialDelayMs).toBe(1000);
});

test("Retries constructor", async () => {
  const r = new Retries({
    maxRetries: 2,
    backoffCoefficient: 2.0,
    initialDelayMs: 2000,
    maxDelayMs: 5000,
  });
  expect(r.maxRetries).toBe(2);
  expect(r.backoffCoefficient).toBe(2.0);
  expect(r.initialDelayMs).toBe(2000);
  expect(r.maxDelayMs).toBe(5000);

  expect(() => new Retries({ maxRetries: -1 })).toThrow(/maxRetries/);
  expect(() => new Retries({ maxRetries: 0, backoffCoefficient: 0.9 })).toThrow(
    /backoffCoefficient/,
  );
  expect(() => new Retries({ maxRetries: 0, initialDelayMs: 61_000 })).toThrow(
    /initialDelayMs/,
  );
  expect(() => new Retries({ maxRetries: 0, maxDelayMs: 500 })).toThrow(
    /maxDelayMs/,
  );
});
