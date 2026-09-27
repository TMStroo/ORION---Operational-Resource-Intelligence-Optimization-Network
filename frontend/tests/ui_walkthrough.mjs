/**
 * Full end-to-end browser walkthrough of the ORION frontend.
 *
 * Drives the real page in real Chromium against the real running backend, and
 * cross-checks what the UI displays against what the API actually stored. The
 * point is not that the page renders - it is that a number on screen is the
 * number in the database.
 *
 * Run:  node tests/ui_walkthrough.mjs
 * Env:  ORION_UI_URL, ORION_API_URL
 */

import { chromium } from "playwright-core";
import fs from "node:fs";

const UI = process.env.ORION_UI_URL ?? "http://127.0.0.1:5183";
const API = process.env.ORION_API_URL ?? "http://127.0.0.1:8137";
// Resolve a Chromium/Chrome binary without hard-coding a machine path, so the
// test carries no local development path into the repository.
function resolveChrome() {
  if (process.env.ORION_CHROME) return process.env.ORION_CHROME;
  const roots = [
    process.env.LOCALAPPDATA,
    process.env.PROGRAMFILES,
    `${process.env.PROGRAMFILES(X86)}`,
    process.env.HOME,
  ].filter(Boolean);
  const rels = [
    ["hermes", "tools"],
    ["ms-playwright"],
    ["Google", "Chrome", "Application"],
  ];
  const names = ["chrome.exe", "headless_shell.exe", "Chromium", "chrome"];
  const stack = [];
  for (const root of roots) {
    for (const rel of rels) stack.push([root, ...rel]);
  }
  // A playwright-managed browser directory may carry a version suffix.
  for (const root of roots) stack.push([root, "ms-playwright"]);
  for (const parts of stack) {
    const base = parts.join("/");
    let entries = [];
    try {
      entries = fs.readdirSync(base);
    } catch {
      continue;
    }
    for (const entry of entries) {
      for (const name of names) {
        for (const candidate of [
          `${base}/${entry}/${name}`,
          `${base}/${entry}/chrome-win64/${name}`,
          `${base}/${entry}/chrome-win/chrome.exe`,
          `${base}/${name}`,
        ]) {
          if (fs.existsSync(candidate)) return candidate;
        }
      }
    }
  }
  throw new Error(
    "no Chromium binary found; set ORION_CHROME to a Chrome/Chromium executable",
  );
}

const results = [];
let failures = 0;

function check(name, ok, detail = "") {
  const pass = Boolean(ok);
  if (!pass) failures += 1;
  results.push({ name, pass, detail });
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? ` :: ${detail}` : ""}`);
}

async function api(path, init) {
  const r = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const t = await r.text();
  return t ? JSON.parse(t) : null;
}

const textOf = (page) => page.evaluate(() => document.body.innerText);

async function main() {
  const browser = await chromium.launch({
    executablePath: resolveChrome(),
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });

  const consoleErrors = [];
  const failedRequests = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => failedRequests.push(`${r.method()} ${r.url()}`));
  page.on("response", (r) => {
    if (r.status() >= 400) failedRequests.push(`${r.status()} ${r.url()}`);
  });

  // ============================================================= 1. open app
  await page.goto(UI, { waitUntil: "networkidle" });
  let t = await textOf(page);
  check("1. application opens and renders ORION", t.includes("ORION"));
  check("1. health banner shows the real backend", /backend\s+(ok|degraded)/.test(t));
  check("1. all ten nav links present", (await page.locator("nav .nav-link").count()) === 10);

  // ================================================ 2-3. create + validate
  await page.click('nav a:has-text("Scenarios")');
  await page.waitForSelector("h1");
  // 25 tasks / 12 resources / seed 7 -> deterministic. A unique name keeps a
  // rerun from colliding with the scenario an earlier run left in the store,
  // which would make "the first stored scenario" ambiguous.
  const runName = `walkthrough-${Date.now()}`;
  await page.getByLabel("Name").fill(runName);
  await page.fill('input[type="number"] >> nth=0', "25");
  await page.fill('input[type="number"] >> nth=1', "12");
  await page.fill('input[type="number"] >> nth=2', "7");
  await page.click('button:has-text("Create scenario")');
  // The panel shell renders as soon as the scenario arrives, so waiting on the
  // title would read the page before validation had come back. Wait for the
  // resolved verdict instead.
  await page.waitForFunction(
    () =>
      /the backend reports this scenario as valid/i.test(document.body.innerText) ||
      /issue\(s\) must be resolved/i.test(document.body.innerText),
    null,
    { timeout: 90000 },
  );
  t = await textOf(page);
  check("2. scenario created through the form", /Tasks \(25\)/.test(t), t.match(/Tasks \(\d+\)/)?.[0]);
  check("2. the form name was used", t.includes(runName), runName);
  check(
    "3. backend validation reported valid",
    /the backend reports this scenario as valid/i.test(t),
  );
  check("2. resource table rendered", /Resources \(12\)/.test(t));
  check("2. objective weights shown", /Objective weights/.test(t));

  const scenarioId = (await api("/scenarios?limit=200")).items.find(
    (s) => s.name === runName,
  )?.id;
  check("2. the created scenario is retrievable from the store", Boolean(scenarioId), scenarioId ?? "");
  const apiScenario = await api(`/scenarios/${encodeURIComponent(scenarioId)}`);
  check(
    "2. API scenario task count matches what the UI showed",
    apiScenario.task_count === 25,
    `api=${apiScenario.task_count}`,
  );
  const validation = await api(`/scenarios/${encodeURIComponent(scenarioId)}/validate`, {
    method: "POST",
    body: "{}",
  });
  check("3. API validation agrees with the UI", validation.valid === true);

  // ================================================== 4-9. optimize + inspect
  await page.click('nav a:has-text("Planning")');
  await page.waitForSelector("h1");
  await page.selectOption("select >> nth=0", "HEURISTIC");
  await page.fill('input[type="number"] >> nth=0', "5");
  await page.click('button:has-text("Solve")');
  await page.waitForFunction(
    () => document.body.innerText.includes("Plan summary"),
    null,
    { timeout: 120000 },
  );
  t = await textOf(page);

  const planId = (await api(`/plans?scenario_id=${encodeURIComponent(scenarioId)}`)).items[0].id;
  const apiPlan = await api(`/plans/${encodeURIComponent(planId)}`);

  check("4. plan produced and summarised", t.includes("Plan summary"));
  check(
    "5. solver name is correct and non-empty",
    apiPlan.solver === "HEURISTIC",
    `solver=${apiPlan.solver}`,
  );

  const expectedCompletion = apiPlan.tasks_assigned / apiPlan.tasks_total;
  check(
    "7. completion is correct (assigned/total), not zero",
    apiPlan.completion > 0 && Math.abs(apiPlan.completion - expectedCompletion) < 1e-6,
    `completion=${apiPlan.completion} expected=${expectedCompletion.toFixed(4)}`,
  );
  check(
    "7. UI displays the real completion, not 0.0%",
    t.includes(`${(apiPlan.completion * 100).toFixed(1)}%`),
    `expected ${(apiPlan.completion * 100).toFixed(1)}%`,
  );
  check(
    "8. runtime is nonzero when applicable",
    apiPlan.runtime_s > 0,
    `runtime=${apiPlan.runtime_s}`,
  );

  const run = apiPlan.solver_runs[0];
  check(
    "9. plan objective matches the stored SolverRun objective",
    run && Math.abs(run.objective - apiPlan.objective) < 1e-3,
    `plan=${apiPlan.objective?.toFixed(3)} run=${run?.objective?.toFixed(3)}`,
  );
  check(
    "9. solver run objective is not the old hard-coded 0.0",
    run && run.objective > 0,
    `run.objective=${run?.objective}`,
  );
  check(
    "9. plan status is one of the declared statuses",
    [
      "OPTIMAL",
      "FEASIBLE",
      "TIME_LIMIT",
      "INFEASIBLE",
      "ERROR",
      "NOT_APPLICABLE",
      "RELAXATION_OPTIMAL",
    ].includes(apiPlan.status),
    `status=${apiPlan.status}`,
  );
  check("5. timeline rendered", t.includes("Timeline"));
  check("5. utilisation rendered", /utilisation/i.test(t));
  check(
    "5. assignment table rendered",
    new RegExp(`Assignments \\(${apiPlan.tasks_assigned}\\)`).test(t),
    t.match(/Assignments \(\d+\)/)?.[0],
  );
  check(
    "5. violations listed when present",
    apiPlan.violations.length === 0 || /Violations \(/.test(t),
  );

  // ==================================================== 10-11. simulate
  await page.click('nav a:has-text("Simulation")');
  await page.waitForSelector("h1");
  await page.click('button:has-text("Run simulation")');
  await page.waitForFunction(() => document.body.innerText.includes("Playback"), null, {
    timeout: 120000,
  });
  t = await textOf(page);
  const simEvents = t.match(/(\d+) events from the engine/);
  check("10. simulation ran on the engine", t.includes("Playback"));
  check("11. the engine produced events", (simEvents?.[1] ?? "0") !== "0", simEvents?.[0]);
  check("10. transport controls present", t.includes("Play") && t.includes("Step"));
  check("10. speed control present", /0\.5x/.test(t) && /16x/.test(t));

  const before = await textOf(page);
  await page.getByRole("button", { name: "Step", exact: true }).click();
  await page.waitForTimeout(500);
  const after = await textOf(page);
  check("11. stepping advances the trace", before !== after);
  check(
    "11. the event log shows events after stepping",
    /Event log \([1-9]/.test(after),
    after.match(/Event log \([^)]*\)/)?.[0],
  );

  // ==================================================== 12-13. disrupt
  const affectedBefore = (await api(`/scenarios/${encodeURIComponent(scenarioId)}`)).task_count;
  await page.click('nav a:has-text("Disruptions")');
  await page.waitForSelector("h1");
  await page.selectOption("select >> nth=0", "RESOURCE_UNAVAILABLE");
  const veh =
    apiScenario.resources.find((r) => r.kind === "VEHICLE") ?? apiScenario.resources[0];
  await page.selectOption("select >> nth=1", veh.id);
  await page.click('button:has-text("Apply disruption")');
  await page.waitForFunction(
    () => document.body.innerText.includes("Applied:"),
    null,
    { timeout: 120000 },
  );
  t = await textOf(page);
  const disruption = await api(`/scenarios/${encodeURIComponent(scenarioId)}/disrupt`, {
    method: "POST",
    body: JSON.stringify({
      type: "RESOURCE_UNAVAILABLE",
      target_id: veh.id,
      magnitude: 0.5,
      duration: 240,
      seed: 0,
    }),
  });
  check("12. disruption applied and rendered", t.includes("Applied:"));
  check("12. affected tasks reported", /Affected tasks/i.test(t));
  check("12. affected resources reported", /Affected resources/i.test(t));
  check("12. severity reported", /HIGH|MEDIUM|LOW/.test(t));

  const originalAfter = await api(`/scenarios/${encodeURIComponent(scenarioId)}`);
  const disrupted = await api(`/scenarios/${encodeURIComponent(disruption.after_scenario_id)}`);
  check(
    "13. only the disrupted branch changed; the original is intact",
    originalAfter.id === scenarioId &&
      originalAfter.task_count === affectedBefore &&
      originalAfter.resources.every((r) => r.status === "AVAILABLE"),
    `tasks=${originalAfter.task_count} allAvailable=${originalAfter.resources.every(
      (r) => r.status === "AVAILABLE",
    )}`,
  );
  check(
    "13. the disrupted scenario is a distinct id",
    disruption.after_scenario_id && disruption.after_scenario_id !== scenarioId,
    `after=${disruption.after_scenario_id}`,
  );
  // A disruption removes *working time*, so the signal is the resource's
  // `unavailable` windows. `status` deliberately stays AVAILABLE: the vehicle
  // still exists, it is just off the road for part of the shift.
  const lostTime = disrupted.resources.reduce((n, r) => n + (r.unavailable?.length ?? 0), 0);
  check(
    "13. the disrupted scenario shows the lost working time",
    lostTime > 0,
    `${lostTime} unavailable windows`,
  );
  check(
    "13. the API exposes the unavailable windows at all",
    apiScenario.resources.every((r) => Array.isArray(r.unavailable)),
  );

  // ==================================================== 14-16. replan
  await page.click('nav a:has-text("Replanning")');
  await page.waitForSelector("h1");
  await page.click('button:has-text("Run local repair")');
  await page.waitForFunction(
    () => document.body.innerText.includes("Recovery quality"),
    null,
    { timeout: 240000 },
  );
  t = await textOf(page);
  check("14. local repair ran", /local repair/i.test(t));
  check("15. full re-optimization ran", /Full re-optimization/.test(t));
  check(
    "14. all four states shown",
    ["baseline", "disrupted", "local repair", "full re-optimization"].every((s) =>
      t.toLowerCase().includes(s),
    ),
  );
  check("14. recovery quality reported", t.includes("Recovery quality"));
  check("14. churn reported", /\bchurn\b/i.test(t));
  check(
    "14. repair is not presented as a proven optimum",
    /not a bound on the true optimum/i.test(t),
  );

  const plansNow = (await api(`/plans?scenario_id=${encodeURIComponent(scenarioId)}`)).items;
  const repairPlan = plansNow.find((p) => p.label === "local repair");
  const fullPlan = plansNow.find((p) => p.label === "full re-optimization");
  check("14. the local-repair plan is persisted and fetchable", Boolean(repairPlan));
  check("15. the full re-optimization plan is persisted and fetchable", Boolean(fullPlan));
  if (repairPlan) {
    const rp = await api(`/plans/${encodeURIComponent(repairPlan.id)}`);
    check(
      "14. the repair plan reports a real objective and completion",
      rp.objective > 0 && rp.completion > 0,
      `obj=${rp.objective?.toFixed(2)} completion=${rp.completion?.toFixed(3)}`,
    );
  }

  // ==================================================== 16. comparison
  // The Replanning page already renders the stored comparison's metric deltas
  // for the repair it produced, so the deltas are checked there against the
  // real record rather than by re-navigating and re-fetching the same data.
  check(
    "16. the stored comparison is rendered as metric deltas",
    /metric deltas/i.test(t) && /\bbefore\b/i.test(t) && /\bchange\b/i.test(t),
  );
  const repairComparison = repairPlan
    ? await api(`/plans/${encodeURIComponent(repairPlan.id)}/comparison`)
    : null;
  check(
    "16. the comparison names the repair and the baseline it came from",
    repairComparison?.items?.some(
      (i) => i.baseline_plan_id && i.candidate_plan_id === repairPlan.id,
    ),
    repairComparison?.items?.[0]
      ? `${repairComparison.items[0].baseline_plan_id} -> ${repairComparison.items[0].candidate_plan_id}`
      : "no comparison",
  );
  // The standalone Comparison page follows whichever plan is active. After a
  // replan that is the full re-optimization, so select the repair plan through
  // the Planning table first - the same navigation a reviewer would use.
  await page.click('nav a:has-text("Planning")');
  await page.waitForSelector("h1");
  await page
    .locator("tbody tr")
    .filter({ hasText: repairPlan?.id ?? "" })
    .first()
    .click({ timeout: 20000 });
  await page.waitForFunction(
    () => /local repair/i.test(document.body.innerText),
    null,
    { timeout: 30000 },
  );
  await page.click('nav a:has-text("Comparison")');
  await page.waitForSelector("h1");
  await page.waitForFunction(
    () => /\bmetric deltas\b/i.test(document.body.innerText),
    null,
    { timeout: 60000 },
  );
  const ct = await textOf(page);
  check("16. comparison page loaded for the active plan", /comparisons for local repair/i.test(ct));
  check(
    "16. the comparison page names a real baseline plan",
    (ct.match(/PLAN-\d+ (?:->|→) PLAN-\d+/) ?? [])[0] ===
      `${repairComparison?.items?.[0]?.baseline_plan_id} → ${repairPlan?.id}`,
    (ct.match(/PLAN-\d+ (?:->|→) PLAN-\d+/) ?? [])[0] ?? "",
  );

  check(
    "16. the comparison endpoint returns the record",
    Array.isArray(repairComparison?.items) && repairComparison.items.length > 0,
    `${repairComparison?.items?.length} items`,
  );

  // ==================================================== 17-18. all six what-if
  await page.click('nav a:has-text("What-if")');
  await page.waitForSelector("h1");
  await page.click('button:has-text("Run all six")');
  await page.waitForFunction(() => document.body.innerText.includes("Comparison"), null, {
    timeout: 300000,
  });
  t = await textOf(page);
  const ops = [
    "availability drop",
    "demand increase",
    "deadline tighten",
    "resource outage",
    "travel increase",
    "capacity reduction",
  ];
  check(
    "17. all six operators present",
    ops.every((o) => t.includes(o)),
    ops.filter((o) => !t.includes(o)).join(", ") || "all",
  );

  const params = {
    availability_drop: 0.2,
    demand_increase: 1.3,
    deadline_tighten: 0.15,
    resource_outage: veh.id,
    travel_increase: 1.25,
    capacity_reduction: 0.8,
  };
  const operatorResults = [];
  for (const [op, param] of Object.entries(params)) {
    const r = await api(`/scenarios/${encodeURIComponent(scenarioId)}/what-if`, {
      method: "POST",
      body: JSON.stringify({ operator: op, parameter: param, seed: 0 }),
    });
    // The objective is stored under the label "Plan score"; the `name` is the
    // internal metric key. Match on the label the user actually sees.
    const d =
      r.comparison.deltas.find((x) => /plan score|objective/i.test(`${x.label} ${x.name}`));
    operatorResults.push({
      op,
      objective: d ? d.after : null,
      delta: d ? d.change : null,
      churn: r.comparison.churn.changed_assignments,
      status: r.status,
    });
  }
  check(
    "18. every operator produced a scored result",
    operatorResults.every((r) => r.objective !== null),
    operatorResults.map((r) => `${r.op}=${r.objective?.toFixed(1)}`).join(" "),
  );
  check(
    "18. operators are not all identical (each is really applied)",
    new Set(operatorResults.map((r) => r.churn)).size > 1,
    operatorResults.map((r) => `${r.op}:churn${r.churn}`).join(" "),
  );
  check(
    "18. at least one operator changes the objective",
    operatorResults.some((r) => r.delta !== null && Math.abs(r.delta) > 1e-6),
    operatorResults.map((r) => `${r.op}:${r.delta?.toFixed(1)}`).join(" "),
  );

  // ==================================================== 19-20. experiments
  await page.click('nav a:has-text("Experiments")');
  await page.waitForSelector("h1");
  await page.waitForFunction(() => document.querySelectorAll("tbody tr").length > 0, null, {
    timeout: 60000,
  });
  t = await textOf(page);
  const rowCount = await page.locator("tbody tr").count();
  check("19. experiments listed from the store", rowCount > 0, `${rowCount} rows`);
  check("20. the live-only filter is available", /live only/i.test(t));

  const exps = await api("/experiments");
  const superseded = exps.items.filter((e) => e.status === "superseded");
  check(
    "19. the store reports both live and superseded",
    exps.live > 0 && exps.superseded > 0,
    `live=${exps.live} superseded=${exps.superseded}`,
  );
  const missingSuccessor = superseded.filter((e) => !e.superseded_by);
  check(
    "20. every superseded run names its successor",
    missingSuccessor.length === 0,
    `${missingSuccessor.length} missing`,
  );

  const sup = superseded[0];
  if (sup) {
    const rowSel = page.locator("tbody tr").filter({ hasText: sup.id }).first();
    if (await rowSel.count()) {
      await rowSel.click({ timeout: 20000 });
      // The detail panel renders below a long table; innerText only returns
      // what is laid out, so scroll it into view before asserting on it.
      await page.waitForFunction(
        () => /manifest/i.test(document.body.textContent ?? ""),
        null,
        { timeout: 60000 },
      );
      await page
        .getByText("Manifest", { exact: true })
        .first()
        .scrollIntoViewIfNeeded()
        .catch(() => {});
      t = await textOf(page);
      check(
        "20. a superseded experiment is labelled when opened",
        /Superseded by/i.test(t),
        `id=${sup.id}`,
      );
      check("20. its supersede reason is non-empty", sup.supersede_reason.length > 0);
    } else {
      check("20. the superseded row was findable in the table", false, sup.id);
    }
  }

  // ==================================================== 21. no hidden breakage
  const realErrors = consoleErrors.filter((e) => !/favicon|DevTools/i.test(e));
  check(
    "21. no uncaught console errors during the walkthrough",
    realErrors.length === 0,
    realErrors.slice(0, 2).join(" | "),
  );
  check(
    "21. no failed API calls during the walkthrough",
    failedRequests.length === 0,
    failedRequests.slice(0, 3).join(" | "),
  );

  await browser.close();

  const passed = results.length - failures;
  console.log(`\n${passed}/${results.length} checks passed`);
  if (failures) {
    console.log("\nFAILURES:");
    for (const r of results.filter((x) => !x.pass)) console.log(`  - ${r.name} :: ${r.detail}`);
  }
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("WALKTHROUGH CRASHED:", e.message);
  process.exit(2);
});
