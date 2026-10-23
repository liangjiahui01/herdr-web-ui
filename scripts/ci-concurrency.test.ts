import { expect, it } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// ci.yml runs on pushes to main (windows-install.yml has no push trigger). A push is grouped by its commit, so a
// later merge never replaces the queued run that says whether the earlier one broke main; a pull request keeps one
// group per branch (per caller workflow) and cancels its superseded run. The values are compared whole, as parsed:
// any edit to them has to change this test too.
type Workflow = { on: { push?: { branches?: string[] } }; concurrency: { group: string; "cancel-in-progress": string } };
const ci = Bun.YAML.parse(readFileSync(join(import.meta.dir, "..", ".github", "workflows", "ci.yml"), "utf8")) as Workflow;

it("groups a push to main by its commit and a pull request by its branch", () => {
  expect(ci.on.push?.branches).toContain("main");
  expect(ci.concurrency).toEqual({
    group: "validation-${{ github.workflow }}-${{ github.event_name == 'push' && github.sha || github.ref }}",
    "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
  });
});
