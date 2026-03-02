import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import importX from "eslint-plugin-import-x";
import { defineConfig, globalIgnores } from "eslint/config";

export default defineConfig([
  globalIgnores(["dist", "docs", "proto"]),
  {
    files: ["**/*.{js,mjs,cjs,ts,mts,cts}"],
    plugins: { js },
    extends: ["js/recommended"],
  },
  {
    files: ["**/*.{js,mjs,cjs,ts,mts,cts}"],
    languageOptions: { globals: globals.node },
  },
  tseslint.configs.recommended,
  {
    files: ["**/*.{js,mjs,cjs,ts,mts,cts}"],
    plugins: { "import-x": importX },
    rules: {
      "import-x/no-extraneous-dependencies": "error",
    },
  },
  {
    files: ["**/*.{ts,mts,cts}"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          args: "all",
          argsIgnorePattern: "^_",
          caughtErrors: "all",
          caughtErrorsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
      // We added this lint because `tsx` gets confused when you export types
      // without using the `type` keyword.
      "@typescript-eslint/consistent-type-exports": "error",
      "object-shorthand": "error",
      "@typescript-eslint/await-thenable": "error",
      "@typescript-eslint/no-deprecated": "error",
      "no-console": "error",
    },
  },
  {
    files: ["test/legacy/**/*.{ts,mts,cts}"],
    rules: {
      "@typescript-eslint/no-deprecated": "off",
    },
  },
  {
    files: ["examples/**/*.{ts,mts,cts}"],
    rules: {
      "no-console": "off",
    },
  },
]);
