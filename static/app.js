let currentDatasetId = null;
let currentProfile = null;
let currentModelId = null;
let currentPredictionApi = null;
let currentDatasetMeta = null;
let datasetRegistry = [];
let sampleDatasets = [];
let dashboardFilters = {};
let lastDashboardData = null;
let currentThreshold = null;
let currentThresholdAnalysis = null;
let selectedFeatureColumns = new Set();
let featureSelectionInitialized = false;
let currentDataQuality = null;
let currentCleaningPlan = null;
let experimentFilterMode = "all";
let modelRegistry = [];
let playgroundMetadata = null;
let playgroundSampleIndex = 0;
let currentDriftReport = null;
let currentMonitoringDashboard = null;
let currentChampionComparison = null;
let currentMonitoringHistory = null;
let currentAlertCenter = null;
let currentApiDocs = null;
let deployActiveTab = "playground";
let selectedDrawerModelId = null;

const statusBox = document.querySelector("#status");
const csvFileInput = document.querySelector("#csvFile");
const dropZone = document.querySelector("#dropZone");
const uploadFormElement = document.querySelector("#uploadForm");
const selectedFileName = document.querySelector("#selectedFileName");
const batchPredictFileInput = document.querySelector("#batchPredictFile");
const batchPredictFileName = document.querySelector("#batchPredictFileName");

function setStatus(text) {
  statusBox.textContent = text;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  if (typeof value !== "number") {
    return escapeHtml(value);
  }
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(value);
}

function formatScoreDelta(value) {
  if (typeof value !== "number") {
    return "First run";
  }
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value)}`;
}

function scoreDeltaClass(value) {
  if (typeof value !== "number" || value === 0) {
    return "neutral";
  }
  return value > 0 ? "positive" : "negative";
}

function buildExperimentQuery() {
  const params = new URLSearchParams({ limit: "18" });
  const target = document.querySelector("#targetColumn")?.value;

  if ((experimentFilterMode === "dataset" || experimentFilterMode === "target") && currentDatasetId) {
    params.set("dataset_id", currentDatasetId);
  }
  if (experimentFilterMode === "target" && target) {
    params.set("target_column", target);
  }

  return params.toString();
}

function updateSidebarCounts() {
  document.querySelector("#sampleCount").textContent = `${sampleDatasets.length} sample CSVs`;
  const workspaceStatus = document.querySelector("#workspaceStatus");
  if (workspaceStatus) {
    workspaceStatus.textContent = currentDatasetId ? `${getDatasetName()} active` : "No active dataset";
  }
  renderSidebarPulse();
  renderStudioOverview();
}

function getBestModelRecord() {
  const target = document.querySelector("#targetColumn")?.value;
  const datasetMatches = currentDatasetId ? modelRegistry.filter((model) => model.dataset_id === currentDatasetId) : [];
  const targetMatches = target ? modelRegistry.filter((model) => model.target_column === target) : [];
  const candidates = datasetMatches.length ? datasetMatches : targetMatches.length ? targetMatches : modelRegistry;
  const scoredModels = candidates
    .filter((model) => typeof model.rank_score === "number")
    .sort((left, right) => right.rank_score - left.rank_score);
  return scoredModels[0] || candidates[0] || null;
}

function getActiveDatasetRecord() {
  return datasetRegistry.find((dataset) => dataset.dataset_id === currentDatasetId) || currentDatasetMeta || null;
}

function datasetCompletenessScore() {
  if (!currentProfile?.columns?.length || !currentDatasetMeta?.rows) {
    return null;
  }
  const rows = Number(currentDatasetMeta.rows || 0);
  const columns = currentProfile.columns || [];
  const columnCount = Number(currentDatasetMeta.columns || columns.length || 0);
  const missingCells = columns.reduce((total, column) => total + Number(column.missing || 0), 0);
  const totalCells = Math.max(rows * columnCount, 1);
  return Math.max(0, Math.min(100, 100 - (missingCells / totalCells) * 100));
}

function currentAlertTotal() {
  if (!currentAlertCenter?.summary) {
    return null;
  }
  const summary = currentAlertCenter.summary;
  return Number(summary.critical || 0) + Number(summary.warning || 0) + Number(summary.info || 0);
}

function overviewModelScore(model) {
  if (!model || typeof model.rank_score !== "number") {
    return null;
  }
  return Math.max(0, Math.min(100, model.rank_score > 1 ? model.rank_score : model.rank_score * 100));
}

function renderSidebarPulse() {
  const container = document.querySelector("#sidebarPulse");
  if (!container) {
    return;
  }
  const bestModel = getBestModelRecord();
  const alertTotal = currentAlertTotal();
  const healthScore = typeof currentMonitoringDashboard?.score === "number" ? currentMonitoringDashboard.score : null;
  const datasetState = currentDatasetId ? "good" : "idle";
  const modelState = bestModel ? "good" : currentDatasetId ? "warn" : "idle";
  const deployState = !bestModel ? "idle" : alertTotal > 0 ? "warn" : "good";

  container.innerHTML = `
    <div class="pulse-item ${datasetState}">
      <span>Dataset</span>
      <strong>${currentDatasetId ? escapeHtml(getDatasetName()) : "Waiting"}</strong>
    </div>
    <div class="pulse-item ${modelState}">
      <span>Models</span>
      <strong>${formatNumber(modelRegistry.length)}</strong>
    </div>
    <div class="pulse-item ${deployState}">
      <span>Deploy</span>
      <strong>${bestModel ? (healthScore === null ? "Ready" : `${formatNumber(healthScore)}/100`) : "Not ready"}</strong>
    </div>
  `;
}

function renderStudioOverview() {
  const overview = document.querySelector("#studioOverview");
  const actionStrip = document.querySelector("#overviewActionStrip");
  if (!overview || !actionStrip) {
    return;
  }

  const dataset = getActiveDatasetRecord();
  const bestModel = getBestModelRecord();
  const completeness = datasetCompletenessScore();
  const modelScore = overviewModelScore(bestModel);
  const healthScore = typeof currentMonitoringDashboard?.score === "number" ? currentMonitoringDashboard.score : null;
  const alertTotal = currentAlertTotal();
  const target = document.querySelector("#targetColumn")?.value || bestModel?.target_column || "Not selected";
  const problem = document.querySelector("#problemType")?.value || bestModel?.problem_type || "Auto";
  const rows = Number(currentDatasetMeta?.rows || dataset?.rows || 0);
  const columns = Number(currentDatasetMeta?.columns || dataset?.columns || currentProfile?.columns?.length || 0);
  const stage = !currentDatasetId
    ? "Upload data"
    : !bestModel
      ? "Train first model"
      : alertTotal > 0
        ? "Review monitoring"
        : "Production ready";

  if (!currentDatasetId && !dataset) {
    overview.className = "studio-overview empty-state";
    overview.textContent = "Load data to build a live studio overview.";
    actionStrip.className = "overview-action-strip empty-state";
    actionStrip.innerHTML = `
      <button type="button" data-overview-action="data">Open data workspace</button>
    `;
    attachOverviewActionEvents();
    return;
  }

  overview.className = "studio-overview";
  overview.innerHTML = `
    <article class="overview-card overview-primary">
      <div class="overview-label">Studio status</div>
      <h2>${escapeHtml(stage)}</h2>
      <p>${escapeHtml(getDatasetName())} / ${formatNumber(rows)} rows / ${formatNumber(columns)} columns</p>
      <div class="overview-meter" aria-label="Dataset completeness"><i style="width: ${formatNumber(completeness ?? 0)}%"></i></div>
      <small>${completeness === null ? "Completeness will update after profiling." : `${formatNumber(completeness)}% dataset completeness`}</small>
    </article>
    <article class="overview-card">
      <span>Target</span>
      <strong>${escapeHtml(target)}</strong>
      <small>${escapeHtml(problem)} problem setup</small>
    </article>
    <article class="overview-card">
      <span>Best model</span>
      <strong>${escapeHtml(bestModel?.best_model || "Not trained")}</strong>
      <small>${bestModel ? `${escapeHtml(bestModel.primary_metric || "score")} ${formatNumber(bestModel.rank_score ?? "n/a")}` : "Train models to fill the leaderboard."}</small>
      <div class="overview-mini-meter"><i style="width: ${formatNumber(modelScore ?? 0)}%"></i></div>
    </article>
    <article class="overview-card">
      <span>Deployment</span>
      <strong>${healthScore === null ? (bestModel ? "Ready to check" : "Locked") : `${formatNumber(healthScore)}/100`}</strong>
      <small>${alertTotal === null ? "Alerts load with monitoring." : `${formatNumber(alertTotal)} active alert${alertTotal === 1 ? "" : "s"}`}</small>
    </article>
  `;

  const actions = [];
  actions.push({ label: "Open data", action: "data", variant: "ghost-button" });
  if (currentDatasetId) {
    actions.push({ label: lastDashboardData ? "Refresh dashboard" : "Generate dashboard", action: "dashboard-load", variant: "" });
    actions.push({ label: bestModel ? "Compare models" : "Train model", action: bestModel ? "registry" : "model", variant: "ghost-button" });
  }
  if (bestModel) {
    actions.push({ label: alertTotal > 0 ? "Review alerts" : "Open deploy", action: alertTotal > 0 ? "deploy-monitoring" : "api", variant: "ghost-button" });
  }

  actionStrip.className = "overview-action-strip";
  actionStrip.innerHTML = actions
    .map((item) => `<button type="button" class="${escapeHtml(item.variant)}" data-overview-action="${escapeHtml(item.action)}">${escapeHtml(item.label)}</button>`)
    .join("");
  attachOverviewActionEvents();
}

function attachOverviewActionEvents() {
  document.querySelectorAll("[data-overview-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.dataset.overviewAction;
      if (action === "dashboard-load") {
        if (currentDatasetId) {
          await loadDashboard();
        } else {
          showView("dashboard");
        }
        return;
      }
      if (action === "deploy-monitoring") {
        showView("api");
        showDeployTab("monitoring");
        await loadModelMonitoring();
        return;
      }
      showView(action);
    });
  });
}

function updateActiveTargetNote() {
  const target = document.querySelector("#targetColumn")?.value;
  const problem = document.querySelector("#problemType")?.value;
  document.querySelector("#activeTargetNote").textContent = target ? `${target} (${problem})` : "Choose a dataset";
  const targetLabel = document.querySelector("#targetColumnLabel");
  const problemLabel = document.querySelector("#problemTypeLabel");
  if (targetLabel) {
    targetLabel.textContent = target || "Select target";
  }
  if (problemLabel) {
    problemLabel.textContent = problem || "Auto detect";
  }
  renderStudioOverview();
}

function setBusy(isBusy) {
  document.body.classList.toggle("busy", isBusy);
}

function showToast(title, detail) {
  let toast = document.querySelector("#studioToast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "studioToast";
    toast.className = "studio-toast";
    document.body.appendChild(toast);
  }

  toast.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span>`;
  toast.classList.add("visible");
  window.clearTimeout(showToast.timeoutId);
  showToast.timeoutId = window.setTimeout(() => {
    toast.classList.remove("visible");
  }, 3600);
}

async function readErrorMessage(response) {
  try {
    const body = await response.json();
    return body.detail || JSON.stringify(body);
  } catch {
    return await response.text();
  }
}

function showView(id, activeButton = null) {
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === id);
  });
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.classList.toggle("active", activeButton ? button === activeButton : button.dataset.view === id);
  });
  const activeStepByView = {
    data: "1",
    dashboard: "3",
    model: "4",
    registry: "5",
    api: "6",
  };
  document.querySelectorAll(".studio-stepper .workflow-step").forEach((step) => {
    step.classList.toggle("active", step.querySelector("span")?.textContent === activeStepByView[id]);
  });
  if (id === "api") {
    showDeployTab(deployActiveTab);
  }
}

function showDeployTab(tabId = "playground") {
  deployActiveTab = tabId;
  document.querySelectorAll("[data-deploy-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.deployTab === tabId);
  });
  document.querySelectorAll("[data-deploy-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.deployPanel !== tabId;
  });
}

function getSelectedModelRecord(modelId = selectedPlaygroundModelId()) {
  return modelRegistry.find((model) => model.model_id === modelId) || null;
}

function deployRoleLabel(status) {
  const labels = {
    production: "Production",
    champion: "Champion",
    challenger: "Challenger",
    unscored: "Unscored",
  };
  return labels[status] || "Model";
}

function updateDeployCommandBar() {
  const model = getSelectedModelRecord();
  const modelName = document.querySelector("#deployCommandModel");
  const meta = document.querySelector("#deployCommandMeta");
  const role = document.querySelector("#deployCommandRole");
  const health = document.querySelector("#deployCommandHealth");
  const alerts = document.querySelector("#deployCommandAlerts");
  const actionButtons = [
    document.querySelector("#commandRunDrift"),
    document.querySelector("#commandPromote"),
    document.querySelector("#commandDocs"),
  ];

  if (!model) {
    if (modelName) {
      modelName.textContent = "No saved model selected";
    }
    if (meta) {
      meta.textContent = "Train or select a model to unlock deployment actions.";
    }
    if (role) {
      role.className = "deploy-command-badge";
      role.textContent = "Not selected";
    }
    if (health) {
      health.textContent = "--";
    }
    if (alerts) {
      alerts.textContent = "--";
    }
    actionButtons.forEach((button) => {
      if (button) {
        button.disabled = true;
      }
    });
    return;
  }

  const comparisonStatus =
    currentChampionComparison?.status ||
    (model.production?.status === "production" ? "production" : model.champion?.status || "challenger");
  const alertCount = (currentAlertCenter?.summary?.critical || 0) + (currentAlertCenter?.summary?.warning || 0) + (currentAlertCenter?.summary?.info || 0);
  if (modelName) {
    modelName.textContent = model.best_model || "Saved model";
  }
  if (meta) {
    meta.textContent = `${model.target_column || "target"} / ${model.problem_type || "problem"} / ${model.primary_metric || "score"} ${formatNumber(model.rank_score ?? "n/a")}`;
  }
  if (role) {
    role.className = `deploy-command-badge ${comparisonStatus}`;
    role.textContent = deployRoleLabel(comparisonStatus);
  }
  if (health) {
    health.textContent = currentMonitoringDashboard?.score !== undefined ? `${formatNumber(currentMonitoringDashboard.score)}/100` : "--";
  }
  if (alerts) {
    alerts.textContent = currentAlertCenter ? formatNumber(alertCount) : "--";
  }
  actionButtons.forEach((button) => {
    if (button) {
      button.disabled = false;
    }
  });
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view, button));
});

document.querySelectorAll("[data-go-view]").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.goView));
});

document.querySelectorAll("[data-sidebar-target]").forEach((button) => {
  const panel = document.querySelector(`#${button.dataset.sidebarTarget}`);
  button.addEventListener("click", () => {
    if (!panel) {
      return;
    }
    const collapsed = panel.classList.toggle("collapsed");
    button.classList.toggle("collapsed", collapsed);
    button.querySelector("b").textContent = collapsed ? "Show" : "Hide";
  });
});

function getPreferredTarget(profile, fallback = null) {
  if (fallback?.target_column) {
    return fallback.target_column;
  }
  return profile?.target_suggestions?.[0]?.name || "";
}

function getSuggestedProblemType(targetName) {
  const suggestion = currentProfile?.target_suggestions?.find((item) => item.name === targetName);
  if (suggestion) {
    return suggestion.problem_type;
  }

  const column = currentProfile?.columns.find((item) => item.name === targetName);
  if (!column) {
    return "auto";
  }

  return column.type === "numeric" && column.unique > 15 ? "regression" : "classification";
}

function populateDatasetSelectors() {
  const selectors = [document.querySelector("#dashboardDatasetSelect"), document.querySelector("#modelDatasetSelect")];
  selectors.forEach((select) => {
    if (!select) {
      return;
    }

    const previousValue = select.value || currentDatasetId || "";
    select.innerHTML = datasetRegistry.length
      ? datasetRegistry
          .map(
            (dataset) => `
              <option value="${escapeHtml(dataset.dataset_id)}">
                ${escapeHtml(dataset.name)} (${formatNumber(dataset.rows)} rows)
              </option>
            `
          )
          .join("")
      : "<option value=\"\">No datasets yet</option>";
    select.value = datasetRegistry.some((dataset) => dataset.dataset_id === previousValue)
      ? previousValue
      : currentDatasetId || datasetRegistry[0]?.dataset_id || "";
  });
}

function populateDriftDatasetSelect() {
  const select = document.querySelector("#driftDatasetSelect");
  if (!select) {
    return;
  }

  const modelDatasetId = playgroundMetadata?.dataset_id || "";
  const previousValue = select.value || "";
  if (!datasetRegistry.length) {
    select.disabled = true;
    select.innerHTML = "<option value=\"\">No datasets yet</option>";
    return;
  }

  select.disabled = false;
  select.innerHTML = datasetRegistry
    .map((dataset) => {
      const trainingLabel = dataset.dataset_id === modelDatasetId ? " - training" : "";
      return `
        <option value="${escapeHtml(dataset.dataset_id)}">
          ${escapeHtml(dataset.name)} (${formatNumber(dataset.rows)} rows${trainingLabel})
        </option>
      `;
    })
    .join("");

  const preferredDataset =
    datasetRegistry.find((dataset) => dataset.dataset_id === previousValue)?.dataset_id ||
    (currentDatasetId && currentDatasetId !== modelDatasetId ? currentDatasetId : "") ||
    datasetRegistry.find((dataset) => dataset.dataset_id !== modelDatasetId)?.dataset_id ||
    modelDatasetId ||
    datasetRegistry[0]?.dataset_id ||
    "";
  select.value = preferredDataset;
}

function getDatasetName(datasetId = currentDatasetId) {
  if (currentDatasetMeta?.dataset_id === datasetId && currentDatasetMeta.name && currentDatasetMeta.name !== "selected dataset") {
    return currentDatasetMeta.name;
  }
  const dataset = datasetRegistry.find((item) => item.dataset_id === datasetId);
  return dataset?.name || "Active dataset";
}

function setWorkspaceActionState() {
  document.querySelectorAll("[data-requires-dataset]").forEach((button) => {
    button.disabled = !currentDatasetId;
  });
}

function renderQualityAssistant(quality, fallbackHtml) {
  if (!quality?.summary) {
    return fallbackHtml;
  }

  const summary = quality.summary;
  const health = quality.health || {
    score: summary.score,
    grade: summary.grade,
    risk_level: summary.grade,
    status: "Review dataset quality",
    can_train: true,
    blocker_count: 0,
    warning_count: summary.issue_count,
    checks: [],
    issue_breakdown: {},
    recommended_actions: quality.recommended_actions || [],
  };
  const topIssues = (quality.issues || []).slice(0, 4);
  const gradeClass = escapeHtml(health.grade).replaceAll(" ", "-");
  const checkItems = health.checks?.length
    ? health.checks
        .map(
          (check) => `
            <div class="health-check ${escapeHtml(check.status)}">
              <div>
                <strong>${escapeHtml(check.label)}</strong>
                <small>${escapeHtml(check.detail)}</small>
              </div>
              <b>${formatNumber(check.score)}%</b>
            </div>
          `
        )
        .join("")
    : "";
  const actions = health.recommended_actions?.length ? health.recommended_actions : quality.recommended_actions || [];
  const scoreCard = `
    <div class="dataset-health-card ${gradeClass}">
      <div class="health-score-ring" style="--score:${Math.max(0, Math.min(Number(health.score || 0), 100))}">
        <strong>${formatNumber(health.score)}%</strong>
        <span>${escapeHtml(health.risk_level)} risk</span>
      </div>
      <div class="health-score-content">
        <span class="panel-kicker">Dataset Health Score</span>
        <h3>${escapeHtml(health.status)}</h3>
        <p>${escapeHtml(health.grade)} &middot; ${formatNumber(summary.issue_count)} issue(s) &middot; ${health.can_train ? "training allowed" : "fix blockers first"}</p>
        <div class="health-pill-row">
          <span>${formatNumber(health.blocker_count || 0)} blockers</span>
          <span>${formatNumber(health.warning_count || 0)} warnings</span>
          <span>${formatNumber(summary.missing_cells)} missing cells</span>
        </div>
      </div>
    </div>
  `;
  const checkPanel = checkItems ? `<div class="health-check-grid">${checkItems}</div>` : "";
  const actionPanel = actions.length
    ? `<div class="health-action-card">
        <span>Fix before training</span>
        <ul>${actions.slice(0, 4).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
      </div>`
    : "";

  if (!topIssues.length) {
    return `
      ${scoreCard}
      ${checkPanel}
      <div class="quality-item clean">
        <span>Data cleaning assistant</span>
        <strong>No major issues found</strong>
        <small>Ready for profiling, visualization, and model training.</small>
      </div>
      <div class="quality-item">
        <span>Duplicates</span>
        <strong>${formatNumber(summary.duplicate_rows)}</strong>
        <small>Exact duplicate rows</small>
      </div>
      <div class="quality-item">
        <span>Missing cells</span>
        <strong>${formatNumber(summary.missing_cells)}</strong>
        <small>Across the full dataset</small>
      </div>
    `;
  }

  return `
    ${scoreCard}
    ${checkPanel}
    ${actionPanel}
    ${topIssues
      .map(
        (item) => `
          <div class="quality-item issue ${escapeHtml(item.severity)}">
            <span>${escapeHtml(item.severity)} &middot; ${escapeHtml(item.type).replaceAll("_", " ")}</span>
            <strong>${escapeHtml(item.title)}</strong>
            <small>${escapeHtml(item.detail)}</small>
            <small>Action: ${escapeHtml(item.action)}</small>
            ${
              item.columns?.length
                ? `<code>${escapeHtml(item.columns.slice(0, 3).join(", "))}${item.columns.length > 3 ? "..." : ""}</code>`
                : ""
            }
          </div>
        `
      )
      .join("")}
  `;
}

function renderCleaningAssistant(plan = currentCleaningPlan) {
  const container = document.querySelector("#cleaningAssistant");
  if (!container) {
    return;
  }

  if (!currentDatasetId) {
    container.className = "cleaning-assistant empty-state";
    container.textContent = "Cleaning plan will appear after loading data.";
    return;
  }

  if (!plan) {
    container.className = "cleaning-assistant loading";
    container.innerHTML = `
      <div class="cleaning-head">
        <div>
          <span class="panel-kicker">One-click cleaning</span>
          <strong>Analyzing data quality actions</strong>
        </div>
      </div>
    `;
    return;
  }

  const summary = plan.summary || {};
  const steps = plan.steps || [];
  const activeSteps = steps.filter((step) => step.changed);
  const scoreBefore = summary.before_score ?? plan.before_quality?.summary?.score ?? "n/a";
  const scoreAfter = summary.after_score ?? plan.after_quality?.summary?.score ?? "n/a";

  container.className = `cleaning-assistant ${activeSteps.length ? "" : "clean"}`;
  container.innerHTML = `
    <div class="cleaning-head">
      <div>
        <span class="panel-kicker">One-click cleaning</span>
        <strong>${activeSteps.length ? `${formatNumber(activeSteps.length)} action(s) ready` : "Dataset already clean"}</strong>
        <small>${escapeHtml(getDatasetName())}</small>
      </div>
      <div class="cleaning-score">
        <span>Score</span>
        <strong>${formatNumber(scoreBefore)}% &rarr; ${formatNumber(scoreAfter)}%</strong>
      </div>
    </div>
    <div class="cleaning-metrics">
      <span>Rows ${formatNumber(summary.before_rows ?? 0)} &rarr; ${formatNumber(summary.after_rows ?? 0)}</span>
      <span>Columns ${formatNumber(summary.before_columns ?? 0)} &rarr; ${formatNumber(summary.after_columns ?? 0)}</span>
      <span>${formatNumber(summary.values_imputed ?? 0)} imputed</span>
      <span>${formatNumber(summary.values_capped ?? 0)} capped</span>
      <span>${formatNumber(summary.rare_values_grouped ?? 0)} rare grouped</span>
    </div>
    <div class="cleaning-step-list">
      ${steps
        .map(
          (step) => `
            <div class="cleaning-step ${step.changed ? "active" : ""}">
              <span>${step.changed ? "Will apply" : "Checked"}</span>
              <strong>${escapeHtml(step.title)}</strong>
              <small>${escapeHtml(step.detail)}</small>
              <em>${escapeHtml(step.impact)}</em>
            </div>
          `
        )
        .join("")}
    </div>
    <div class="cleaning-actions">
      <button id="refreshCleaningPlan" type="button" class="ghost-button">Refresh plan</button>
      <button id="applyCleaningPlan" type="button" ${plan.apply_available ? "" : "disabled"}>Apply cleaning</button>
    </div>
  `;

  document.querySelector("#refreshCleaningPlan")?.addEventListener("click", loadCleaningPlan);
  document.querySelector("#applyCleaningPlan")?.addEventListener("click", applyCleaningPlan);
}

function cleaningRequestBody() {
  const target = document.querySelector("#targetColumn")?.value || null;
  return { target_column: target || null };
}

async function loadCleaningPlan() {
  if (!currentDatasetId) {
    currentCleaningPlan = null;
    renderCleaningAssistant();
    return;
  }

  const datasetId = currentDatasetId;
  currentCleaningPlan = null;
  renderCleaningAssistant();

  const response = await fetch(`/api/datasets/${encodeURIComponent(datasetId)}/cleaning-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleaningRequestBody()),
  });

  if (datasetId !== currentDatasetId) {
    return;
  }

  if (!response.ok) {
    const container = document.querySelector("#cleaningAssistant");
    if (container) {
      container.className = "cleaning-assistant empty-state";
      container.textContent = await readErrorMessage(response);
    }
    return;
  }

  currentCleaningPlan = await response.json();
  renderCleaningAssistant();
}

async function applyCleaningPlan() {
  if (!currentDatasetId) {
    alert("Load a dataset first.");
    return;
  }

  const sourceDatasetId = currentDatasetId;
  setStatus("Cleaning dataset");
  setBusy(true);
  const response = await fetch(`/api/datasets/${encodeURIComponent(sourceDatasetId)}/clean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleaningRequestBody()),
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Cleaning failed");
    setBusy(false);
    return;
  }

  const result = await response.json();
  renderProfile(result);
  setStatus("Cleaned dataset loaded");
  showToast("Data cleaned", `${formatNumber(result.rows)} rows and ${formatNumber(result.columns)} columns are ready for training.`);
  setBusy(false);
  showView("data");
}


function renderWorkspaceOverview() {
  const title = document.querySelector("#workspaceTitle");
  const subtitle = document.querySelector("#workspaceSubtitle");
  const stats = document.querySelector("#workspaceStats");
  const quality = document.querySelector("#workspaceQuality");
  if (!title || !subtitle || !stats || !quality) {
    return;
  }

  setWorkspaceActionState();
  updateSidebarCounts();

  if (!currentDatasetId || !currentProfile || !currentDatasetMeta) {
    title.textContent = "No dataset loaded";
    subtitle.textContent = "Load a sample or upload a CSV to unlock profiling, visualization, training, and prediction actions.";
    stats.className = "workspace-stat-grid empty-state";
    stats.textContent = "Dataset metrics will appear after loading data.";
    quality.className = "quality-grid empty-state";
    quality.textContent = "Automated quality checks will appear here.";
    currentCleaningPlan = null;
    renderCleaningAssistant();
    return;
  }

  const columns = currentProfile.columns || [];
  const rows = Number(currentDatasetMeta.rows || 0);
  const columnCount = Number(currentDatasetMeta.columns || columns.length || 0);
  const missingCells = columns.reduce((total, column) => total + Number(column.missing || 0), 0);
  const totalCells = Math.max(rows * columnCount, 1);
  const completeness = Math.max(0, 100 - (missingCells / totalCells) * 100);
  const numericCount = columns.filter((column) => column.type === "numeric").length;
  const categoricalCount = columns.filter((column) => column.type === "categorical").length;
  const dateCount = columns.filter((column) => column.type === "date").length;
  const topTarget = currentProfile.target_suggestions?.[0];
  const activeTarget = document.querySelector("#targetColumn")?.value || topTarget?.name || "";
  const problemType = activeTarget ? getSuggestedProblemType(activeTarget) : "auto";

  title.textContent = getDatasetName();
  subtitle.textContent = `${formatNumber(rows)} rows, ${formatNumber(columnCount)} columns. Ready for profiling, dashboarding, model training, and API prediction.`;
  stats.className = "workspace-stat-grid";
  stats.innerHTML = `
    <div><span>Rows</span><strong>${formatNumber(rows)}</strong></div>
    <div><span>Columns</span><strong>${formatNumber(columnCount)}</strong></div>
    <div><span>Complete</span><strong>${formatNumber(completeness)}%</strong></div>
    <div><span>Target</span><strong>${escapeHtml(activeTarget || "Choose")}</strong></div>
  `;

  quality.className = "quality-grid";
  quality.innerHTML = `
    <div class="quality-item">
      <span>Feature mix</span>
      <strong>${formatNumber(numericCount)} numeric &middot; ${formatNumber(categoricalCount)} categorical &middot; ${formatNumber(dateCount)} date</strong>
    </div>
    <div class="quality-item">
      <span>Missing cells</span>
      <strong>${formatNumber(missingCells)}</strong>
    </div>
    <div class="quality-item">
      <span>Suggested target</span>
      <strong>${escapeHtml(topTarget?.name || "Review columns")}</strong>
      ${topTarget ? `<button type="button" data-use-target="${escapeHtml(topTarget.name)}">Use target</button>` : ""}
    </div>
    <div class="quality-item">
      <span>Problem type</span>
      <strong>${escapeHtml(problemType)}</strong>
    </div>
  `;
  quality.innerHTML = renderQualityAssistant(currentDataQuality, quality.innerHTML);

  document.querySelectorAll("[data-use-target]").forEach((button) => {
    button.addEventListener("click", () => {
      const targetSelect = document.querySelector("#targetColumn");
      const problemSelect = document.querySelector("#problemType");
      targetSelect.value = button.dataset.useTarget;
      problemSelect.value = getSuggestedProblemType(button.dataset.useTarget);
      updateActiveTargetNote();
      renderWorkspaceOverview();
      currentCleaningPlan = null;
      renderCleaningAssistant();
      loadCleaningPlan();
      loadModelSuggestions();
      showToast("Target applied", `${button.dataset.useTarget} is now the active training target.`);
    });
  });
}

function renderSampleCards(samples) {
  const rail = document.querySelector("#sampleDatasetRail");
  const insight = document.querySelector("#sampleInsightList");
  if (!samples.length) {
    rail.className = "dataset-rail empty-state";
    rail.textContent = "No sample CSVs found in sample_data.";
    insight.innerHTML = "";
    return;
  }

  rail.className = "dataset-rail";
  rail.innerHTML = samples
    .map(
      (sample) => `
        <article class="source-card sample-card" data-sample-id="${escapeHtml(sample.sample_id)}">
          <span>${escapeHtml(sample.domain.slice(0, 2).toUpperCase())}</span>
          <strong>${escapeHtml(sample.title)}</strong>
          <small>${escapeHtml(sample.filename)} &middot; ${formatNumber(sample.rows)} rows</small>
          <div class="drag-hint">Drag to bucket</div>
          <button class="small-action" type="button" data-load-sample="${escapeHtml(sample.sample_id)}">Load</button>
        </article>
      `
    )
    .join("");

  insight.innerHTML = samples
    .slice(0, 3)
    .map(
      (sample) => `
        <div>
          <strong>${escapeHtml(sample.title)}</strong>
          <span>${escapeHtml(sample.problem_type || "auto")} target: ${escapeHtml(sample.target_column || "suggested")}</span>
        </div>
      `
    )
    .join("");

  document.querySelectorAll("[data-load-sample]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      await loadSampleDataset(button.dataset.loadSample);
    });
  });

  document.querySelectorAll(".sample-card").forEach((card) => {
    card.addEventListener("click", async () => loadSampleDataset(card.dataset.sampleId));
    card.setAttribute("draggable", "true");
    card.addEventListener("dragstart", (event) => {
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("application/x-automate-sample", card.dataset.sampleId);
      event.dataTransfer.setData("text/plain", `sample:${card.dataset.sampleId}`);
      card.classList.add("dragging");
      dropZone.classList.add("bucket-ready");
      setStatus("Drag sample into upload bucket");
    });
    card.addEventListener("dragend", () => {
      card.classList.remove("dragging");
      dropZone.classList.remove("bucket-ready");
    });
  });
}

async function loadSampleDatasets() {
  const response = await fetch("/api/sample-datasets");
  if (!response.ok) {
    document.querySelector("#sampleDatasetRail").textContent = await readErrorMessage(response);
    return;
  }

  const body = await response.json();
  sampleDatasets = body.samples || [];
  renderSampleCards(sampleDatasets);
  updateSidebarCounts();
}

async function loadSampleDataset(sampleId) {
  setStatus("Loading sample CSV");
  setBusy(true);
  const response = await fetch(`/api/sample-datasets/${encodeURIComponent(sampleId)}/load`, {
    method: "POST",
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Sample load failed");
    setBusy(false);
    return;
  }

  const body = await response.json();
  renderProfile(body);
  if (body.suggested_target?.target_column) {
    document.querySelector("#targetColumn").value = body.suggested_target.target_column;
    document.querySelector("#problemType").value = body.suggested_target.problem_type || getSuggestedProblemType(body.suggested_target.target_column);
    updateActiveTargetNote();
    loadModelSuggestions();
  }
  setStatus(`${body.sample?.title || body.filename} loaded`);
  showToast("Upload complete", `${body.sample?.title || body.filename} profile is ready.`);
  setBusy(false);
  showView("data");
}

function renderProfile(response) {
  currentDatasetId = response.dataset_id;
  currentProfile = response.profile;
  currentDataQuality = response.quality || null;
  currentCleaningPlan = null;
  selectedFeatureColumns = new Set();
  featureSelectionInitialized = false;
  currentDatasetMeta = {
    dataset_id: response.dataset_id,
    name: response.sample?.title || response.filename || response.table || "Active dataset",
    rows: response.rows,
    columns: response.columns,
    source: response.source || (response.sample ? "sample" : response.table ? "sqlite" : "csv"),
  };
  setStatus(`${response.rows} rows loaded`);
  dashboardFilters = {};

  document.querySelector("#profileSummary").innerHTML = `
    <div class="stat"><span>Rows</span><strong>${formatNumber(response.rows)}</strong></div>
    <div class="stat"><span>Columns</span><strong>${formatNumber(response.columns)}</strong></div>
    <div class="stat"><span>Dataset ID</span><strong>${escapeHtml(response.dataset_id.slice(0, 8))}</strong></div>
    <div class="stat"><span>Source</span><strong>${escapeHtml(response.filename || response.table || "database")}</strong></div>
  `;

  const suggestedNames = new Set((currentProfile.target_suggestions || []).map((item) => item.name));
  const orderedColumns = [
    ...currentProfile.columns.filter((column) => suggestedNames.has(column.name)),
    ...currentProfile.columns.filter((column) => !suggestedNames.has(column.name)),
  ];
  const preferredTarget = getPreferredTarget(currentProfile, response.suggested_target);
  const options = orderedColumns
    .map((column) => `<option value="${escapeHtml(column.name)}">${escapeHtml(column.name)}</option>`)
    .join("");
  document.querySelector("#targetColumn").innerHTML = options;
  if (preferredTarget) {
    document.querySelector("#targetColumn").value = preferredTarget;
  }
  updateProblemTypeSuggestion();

  const rows = currentProfile.columns
    .map(
      (column) => `
        <tr>
          <td>${escapeHtml(column.name)}${suggestedNames.has(column.name) ? " <span class=\"target-chip\">Target</span>" : ""}</td>
          <td><span class="type-badge ${column.type}">${column.type}</span></td>
          <td>${formatNumber(column.missing)}</td>
          <td>${column.missing_percent}%</td>
          <td>${formatNumber(column.unique)}</td>
        </tr>
      `
    )
    .join("");

  document.querySelector("#profileTable").innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Missing</th>
          <th>Missing %</th>
          <th>Unique</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  populateDatasetSelectors();
  updateActiveTargetNote();
  renderFeatureSelector();
  renderWorkspaceOverview();
  renderCleaningAssistant();
  loadDatasetRegistry();
  loadDataPreview(currentDatasetId);
  loadCleaningPlan();
}

async function selectDataset(datasetId) {
  if (!datasetId) {
    return;
  }

  currentDatasetId = datasetId;
  dashboardFilters = {};
  setStatus(`Dataset ${currentDatasetId.slice(0, 8)} selected`);
  populateDatasetSelectors();
  await loadDatasetProfile(currentDatasetId);
}

async function loadDatasetRegistry() {
  const response = await fetch("/api/datasets");
  if (!response.ok) {
    return;
  }

  const body = await response.json();
  datasetRegistry = body.datasets || [];
  populateDatasetSelectors();
  populateDriftDatasetSelect();
  updateSidebarCounts();
  renderWorkspaceOverview();
}

async function loadDataPreview(datasetId) {
  if (!datasetId) {
    return;
  }

  const container = document.querySelector("#dataPreview");
  const response = await fetch(`/api/datasets/${datasetId}/preview?limit=20`);
  if (!response.ok) {
    container.className = "table-wrap empty-state";
    container.textContent = await readErrorMessage(response);
    return;
  }

  const preview = await response.json();
  container.className = "table-wrap";
  container.innerHTML = `
    <table>
      <thead>
        <tr>${preview.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${preview.rows
          .map(
            (row) => `
              <tr>${preview.columns.map((column) => `<td>${escapeHtml(row[column] ?? "")}</td>`).join("")}</tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

async function loadDatasetProfile(datasetId) {
  const response = await fetch(`/api/datasets/${datasetId}/profile`);
  if (!response.ok) {
    return;
  }

  const body = await response.json();
  const registryRecord = datasetRegistry.find((dataset) => dataset.dataset_id === datasetId);
  renderProfile({
    dataset_id: datasetId,
    filename: registryRecord?.name || "selected dataset",
    rows: body.rows,
    columns: body.columns,
    profile: body.profile,
    quality: body.quality,
  });
}

function riskClass(value) {
  return ["low", "medium", "high"].includes(value) ? value : "unknown";
}

function trustBadge(model) {
  const trust = model.trust_score || {};
  if (typeof trust.score !== "number") {
    return `<span class="trust-badge unknown">n/a</span>`;
  }
  return `
    <span class="trust-badge ${riskClass(trust.risk_level)}">
      ${formatNumber(trust.score)}/100 &middot; ${escapeHtml(trust.trust_label || "Risk")}
    </span>
  `;
}

function championBadge(model) {
  const production = model.production || {};
  const champion = model.champion || {};
  const status = production.status === "production" ? "production" : champion.status || "unscored";
  const rank = champion.rank ? `#${formatNumber(champion.rank)}` : "n/a";
  const delta = typeof champion.score_delta_vs_champion === "number"
    ? ` ${formatScoreDelta(champion.score_delta_vs_champion)}`
    : "";
  return `
    <span class="champion-badge ${escapeHtml(status)}">
      ${escapeHtml(status)} ${escapeHtml(rank)}${escapeHtml(delta)}
    </span>
  `;
}

function metricTiles(metrics, emptyText = "Metrics unavailable") {
  const entries = Object.entries(metrics || {}).slice(0, 8);
  if (!entries.length) {
    return `<div class="drawer-empty">${escapeHtml(emptyText)}</div>`;
  }
  return entries
    .map(
      ([key, value]) => `
        <div class="drawer-metric">
          <span>${escapeHtml(key.replaceAll("_", " "))}</span>
          <strong>${formatNumber(value)}</strong>
        </div>
      `
    )
    .join("");
}

function renderDrawerFeatures(features) {
  const visibleFeatures = (features || []).slice(0, 6);
  if (!visibleFeatures.length) {
    return `<div class="drawer-empty">Feature importance is not available for this model.</div>`;
  }
  return visibleFeatures
    .map((feature) => {
      const share = typeof feature.share === "number" ? feature.share * 100 : Number(feature.importance || 0) * 100;
      const width = Math.max(4, Math.min(100, share || 0));
      return `
        <div class="drawer-feature">
          <div>
            <strong>${escapeHtml(feature.feature || "Feature")}</strong>
            <span>${escapeHtml(feature.strength || feature.impact || "Signal")}</span>
          </div>
          <b>${formatNumber(feature.importance ?? feature.share ?? "n/a")}</b>
          <i><em style="width:${formatNumber(width)}%"></em></i>
        </div>
      `;
    })
    .join("");
}

function renderDrawerActions(actions) {
  if (!actions?.length) {
    return `<li class="drawer-empty">No saved recommendations for this run.</li>`;
  }
  return actions
    .slice(0, 4)
    .map((action) => `<li>${escapeHtml(action)}</li>`)
    .join("");
}

function openModelDrawer(modelId) {
  const model = modelRegistry.find((item) => item.model_id === modelId);
  const drawer = document.querySelector("#modelDetailDrawer");
  const backdrop = document.querySelector("#modelDrawerBackdrop");
  const title = document.querySelector("#modelDrawerTitle");
  const subtitle = document.querySelector("#modelDrawerSubtitle");
  const body = document.querySelector("#modelDetailBody");
  const deployButton = document.querySelector("#drawerDeployModel");
  if (!model || !drawer || !backdrop || !title || !subtitle || !body) {
    return;
  }

  selectedDrawerModelId = model.model_id;
  const productionStatus = model.production?.status === "production" ? "Production" : model.champion?.status || "Saved model";
  title.textContent = model.best_model || "Saved model";
  subtitle.textContent = `${model.target_column || "target"} / ${model.problem_type || "problem"} / ${model.primary_metric || "score"} ${formatNumber(model.rank_score ?? "n/a")}`;
  if (deployButton) {
    deployButton.disabled = false;
  }

  body.className = "model-detail-body";
  body.innerHTML = `
    <section class="drawer-status">
      <div>
        <span>Role</span>
        ${championBadge(model)}
      </div>
      <div>
        <span>Risk</span>
        ${trustBadge(model)}
      </div>
      <div>
        <span>Quality</span>
        <strong>${escapeHtml(model.quality_label || productionStatus)}</strong>
      </div>
    </section>

    <section class="drawer-section">
      <div class="drawer-section-head">
        <span>Holdout metrics</span>
        <b>${escapeHtml(model.primary_metric || "score")}</b>
      </div>
      <div class="drawer-metric-grid">${metricTiles(model.holdout_metrics)}</div>
    </section>

    <section class="drawer-section">
      <div class="drawer-section-head">
        <span>Training footprint</span>
        <b>${formatNumber(model.trained_models ?? 0)}/${formatNumber(model.candidate_models ?? 0)} trained</b>
      </div>
      <div class="drawer-metric-grid">
        <div class="drawer-metric"><span>Model features</span><strong>${formatNumber(model.model_feature_count ?? "n/a")}</strong></div>
        <div class="drawer-metric"><span>Raw features</span><strong>${formatNumber(model.raw_feature_count ?? "n/a")}</strong></div>
        <div class="drawer-metric"><span>Failed models</span><strong>${formatNumber(model.failed_models ?? 0)}</strong></div>
        <div class="drawer-metric"><span>Threshold</span><strong>${formatNumber(model.recommended_threshold ?? "n/a")}</strong></div>
      </div>
    </section>

    <section class="drawer-section">
      <div class="drawer-section-head">
        <span>Top features</span>
        <b>${formatNumber((model.top_features || []).length)}</b>
      </div>
      <div class="drawer-feature-list">${renderDrawerFeatures(model.top_features)}</div>
    </section>

    <section class="drawer-section">
      <div class="drawer-section-head">
        <span>Next actions</span>
        <b>Recommended</b>
      </div>
      <ul class="drawer-action-list">${renderDrawerActions(model.next_actions)}</ul>
    </section>

    <section class="drawer-section">
      <div class="drawer-section-head">
        <span>Prediction API</span>
        <b>${escapeHtml(model.model_id?.slice(0, 8) || "model")}</b>
      </div>
      <code class="drawer-api">${escapeHtml(model.prediction_api || "No endpoint saved")}</code>
    </section>
  `;

  backdrop.hidden = false;
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.body.classList.add("drawer-open");
}

function closeModelDrawer() {
  const drawer = document.querySelector("#modelDetailDrawer");
  const backdrop = document.querySelector("#modelDrawerBackdrop");
  if (drawer) {
    drawer.classList.remove("open");
    drawer.setAttribute("aria-hidden", "true");
  }
  if (backdrop) {
    backdrop.hidden = true;
  }
  document.body.classList.remove("drawer-open");
}

async function openDrawerModelInDeploy() {
  if (!selectedDrawerModelId) {
    return;
  }
  closeModelDrawer();
  showView("api");
  showDeployTab("playground");
  const select = document.querySelector("#playgroundModelSelect");
  if (select && [...select.options].some((option) => option.value === selectedDrawerModelId)) {
    select.value = selectedDrawerModelId;
    await loadPredictionPlayground(selectedDrawerModelId);
  }
}

function modelRegistryTable(models, limit = null) {
  const rows = (limit ? models.slice(0, limit) : models)
    .map(
      (model) => `
        <tr>
          <td>${escapeHtml(model.best_model)}</td>
          <td>${escapeHtml(model.target_column)}</td>
          <td><span class="type-badge ${model.problem_type}">${model.problem_type}</span></td>
          <td><span class="metric-pill">${escapeHtml(model.quality_label || "saved")}</span></td>
          <td>${championBadge(model)}</td>
          <td>${trustBadge(model)}</td>
          <td>${escapeHtml(model.primary_metric)}</td>
          <td>${formatNumber(model.rank_score ?? "")}</td>
          <td>${formatNumber(model.trained_models ?? "")}/${formatNumber(model.candidate_models ?? "")}</td>
          <td>${formatNumber(model.model_feature_count ?? "")}/${formatNumber(model.raw_feature_count ?? "")}</td>
          <td><code>${escapeHtml(model.prediction_api)}</code></td>
          <td>${escapeHtml(model.created_at)}</td>
          <td><button class="model-inspect-button ghost-button" type="button" data-inspect-model="${escapeHtml(model.model_id)}">Inspect</button></td>
        </tr>
      `
    )
    .join("");

  return `
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Target</th>
          <th>Problem</th>
          <th>Quality</th>
          <th>Role</th>
          <th>Risk</th>
          <th>Metric</th>
          <th>Score</th>
          <th>Trained</th>
          <th>Features</th>
          <th>Prediction API</th>
          <th>Created</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderTrustScorePanel(models) {
  const container = document.querySelector("#modelTrustScore");
  if (!container) {
    return;
  }

  if (!models.length) {
    container.className = "trust-score empty-state";
    container.textContent = "Train a model to calculate risk.";
    return;
  }

  const model = models[0];
  const trust = model.trust_score || {};
  if (typeof trust.score !== "number") {
    container.className = "trust-score empty-state";
    container.textContent = "Trust score is not available for this model.";
    return;
  }

  const checks = (trust.checks || []).slice(0, 5);
  container.className = `trust-score ${riskClass(trust.risk_level)}`;
  container.innerHTML = `
    <div class="trust-score-head">
      <div>
        <span>${escapeHtml(trust.trust_label || "Risk")}</span>
        <strong>${formatNumber(trust.score)}/100</strong>
      </div>
      <b>${escapeHtml(trust.decision || "Review model")}</b>
    </div>
    <p>${escapeHtml(trust.summary || "Trust score calculated from validation and governance signals.")}</p>
    <div class="trust-meter"><i style="width:${Math.max(3, Math.min(trust.score, 100))}%"></i></div>
    <div class="trust-mini-stats">
      <span><b>${formatNumber(trust.blockers ?? 0)}</b> blockers</span>
      <span><b>${formatNumber(trust.warnings ?? 0)}</b> warnings</span>
      <span><b>${escapeHtml(model.best_model || "Model")}</b></span>
    </div>
    <div class="trust-check-list">
      ${checks
        .map(
          (check) => `
            <div class="trust-check ${escapeHtml(check.status)}">
              <strong>${escapeHtml(check.label)}</strong>
              <span>${escapeHtml(check.detail)}</span>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderRegistryStats(models) {
  const stats = document.querySelector("#registryStats");
  const apiCard = document.querySelector("#registryApiCard");
  if (!stats || !apiCard) {
    return;
  }
  document.querySelector("#modelCount").textContent = `${models.length} trained models`;

  if (!models.length) {
    stats.className = "summary-row compact empty-state";
    stats.textContent = "Train a model to build registry stats.";
    apiCard.textContent = "No model endpoint yet.";
    renderTrustScorePanel(models);
    return;
  }

  const classificationCount = models.filter((model) => model.problem_type === "classification").length;
  const regressionCount = models.filter((model) => model.problem_type === "regression").length;
  const latest = models[0];
  stats.className = "summary-row compact";
  stats.innerHTML = `
    <div class="stat"><span>Total models</span><strong>${models.length}</strong></div>
    <div class="stat"><span>Classification</span><strong>${classificationCount}</strong></div>
    <div class="stat"><span>Regression</span><strong>${regressionCount}</strong></div>
    <div class="stat"><span>Latest quality</span><strong>${escapeHtml(latest.quality_label || "saved")}</strong></div>
  `;
  const firstAction = latest.next_actions?.[0];
  apiCard.innerHTML = `
    <strong>${escapeHtml(latest.best_model)}</strong>
    <p class="muted">${escapeHtml(latest.target_column)} &middot; ${escapeHtml(latest.primary_metric)}: ${formatNumber(latest.rank_score ?? "n/a")}</p>
    ${firstAction ? `<p class="muted">${escapeHtml(firstAction)}</p>` : ""}
    <code>${escapeHtml(latest.prediction_api)}</code>
  `;
  renderTrustScorePanel(models);
}

function selectedExportModelId() {
  const select = document.querySelector("#exportModelSelect");
  return select?.value || modelRegistry[0]?.model_id || "";
}

function buildExperimentExportQuery() {
  const params = new URLSearchParams({ limit: "100" });
  const target = document.querySelector("#targetColumn")?.value;

  if ((experimentFilterMode === "dataset" || experimentFilterMode === "target") && currentDatasetId) {
    params.set("dataset_id", currentDatasetId);
  }
  if (experimentFilterMode === "target" && target) {
    params.set("target_column", target);
  }

  return params.toString();
}

function triggerDownload(url) {
  const link = document.createElement("a");
  link.href = url;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function renderExportCenter(models) {
  const select = document.querySelector("#exportModelSelect");
  const body = document.querySelector("#exportCenterBody");
  if (!select || !body) {
    return;
  }

  if (!models.length) {
    select.innerHTML = "";
    select.disabled = true;
    body.className = "export-grid empty-state";
    body.textContent = "Train a model to unlock exports.";
    return;
  }

  const previousSelection = selectedExportModelId();
  select.disabled = false;
  select.innerHTML = models
    .map(
      (model) => `
        <option value="${escapeHtml(model.model_id)}">${escapeHtml(model.best_model || "Model")} &middot; ${escapeHtml(model.target_column || "target")} &middot; ${formatNumber(model.rank_score ?? "n/a")}</option>
      `
    )
    .join("");
  if (previousSelection && models.some((model) => model.model_id === previousSelection)) {
    select.value = previousSelection;
  }
  select.onchange = () => renderExportCenter(modelRegistry);

  const selected = models.find((model) => model.model_id === selectedExportModelId()) || models[0];
  const hasLeaderboard = Boolean(selected?.leaderboard_snapshot?.length || selected?.best_model);
  const hasExplainability = Boolean(selected?.explainability_studio?.features?.length || selected?.top_features?.length);

  body.className = "export-grid";
  body.innerHTML = `
    <article class="export-card">
      <span>Model artifact</span>
      <strong>${escapeHtml(selected.best_model || "Selected model")}</strong>
      <small>Download the trained .joblib model package.</small>
      <button type="button" data-export-action="model">Download model</button>
    </article>
    <article class="export-card">
      <span>Experiment history</span>
      <strong>${escapeHtml(experimentFilterMode.replaceAll("_", " "))}</strong>
      <small>CSV of saved runs, score deltas, best-run flags, and APIs.</small>
      <button type="button" data-export-action="experiments">Export history CSV</button>
    </article>
    <article class="export-card ${hasLeaderboard ? "" : "disabled"}">
      <span>Leaderboard</span>
      <strong>${selected.leaderboard_snapshot?.length ? `${formatNumber(selected.leaderboard_snapshot.length)} models` : "Best model only"}</strong>
      <small>CSV of trained candidates, metrics, CV scores, and errors when available.</small>
      <button type="button" data-export-action="leaderboard" ${hasLeaderboard ? "" : "disabled"}>Export leaderboard</button>
    </article>
    <article class="export-card ${hasExplainability ? "" : "disabled"}">
      <span>Explainability</span>
      <strong>${hasExplainability ? "Feature drivers" : "No explainability saved"}</strong>
      <small>CSV of feature importance, strength, impact, and interpretation.</small>
      <button type="button" data-export-action="explainability" ${hasExplainability ? "" : "disabled"}>Export explainability</button>
    </article>
    <article class="export-card">
      <span>Auto report</span>
      <strong>PPTX report</strong>
      <small>Six-slide project report with model scores, readiness, leaderboard, explainability, and next actions.</small>
      <button type="button" data-export-action="report">Download report</button>
    </article>
  `;

  body.querySelectorAll("[data-export-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const modelId = encodeURIComponent(selectedExportModelId());
      const action = button.dataset.exportAction;
      if (action === "model") {
        triggerDownload(`/api/models/${modelId}/download`);
      }
      if (action === "experiments") {
        triggerDownload(`/api/experiments/export?${buildExperimentExportQuery()}`);
      }
      if (action === "leaderboard") {
        triggerDownload(`/api/models/${modelId}/leaderboard/export`);
      }
      if (action === "explainability") {
        triggerDownload(`/api/models/${modelId}/explainability/export`);
      }
      if (action === "report") {
        triggerDownload(`/api/models/${modelId}/report`);
      }
      setStatus("Export started");
    });
  });
}

function renderExperimentComparison(history) {
  const container = document.querySelector("#experimentComparison");
  if (!container) {
    return;
  }

  const experiments = Array.isArray(history) ? history : history?.experiments || [];
  const summary = Array.isArray(history) ? {} : history?.summary || {};
  const target = document.querySelector("#targetColumn")?.value;
  const filterOptions = [
    { mode: "all", label: "All runs", disabled: false },
    { mode: "dataset", label: "Active dataset", disabled: !currentDatasetId },
    { mode: "target", label: "Current target", disabled: !currentDatasetId || !target },
  ];

  const filterToolbar = `
    <div class="experiment-toolbar">
      ${filterOptions
        .map(
          (option) => `
            <button
              type="button"
              data-experiment-filter="${option.mode}"
              class="ghost-button ${experimentFilterMode === option.mode ? "active" : ""}"
              ${option.disabled ? "disabled" : ""}
            >${escapeHtml(option.label)}</button>
          `
        )
        .join("")}
    </div>
  `;

  if (!experiments.length) {
    container.className = "experiment-grid empty-state";
    container.innerHTML = `${filterToolbar}<p>Train models to compare experiments.</p>`;
    attachExperimentFilterEvents(container);
    return;
  }

  const latest = summary.latest_run || experiments[0];
  const best = summary.best_run || experiments.find((item) => item.is_best_for_group) || experiments[0];
  const recent = experiments.slice(0, 6);
  container.className = "experiment-history";
  container.innerHTML = `
    ${filterToolbar}
    <div class="experiment-summary-strip">
      <div class="stat"><span>Visible runs</span><strong>${formatNumber(summary.filtered_runs ?? experiments.length)}</strong></div>
      <div class="stat"><span>Total saved</span><strong>${formatNumber(summary.total_runs ?? experiments.length)}</strong></div>
      <div class="stat wide"><span>Best run</span><strong>${escapeHtml(best.best_model || "Model")}</strong></div>
      <div class="stat"><span>${escapeHtml(best.primary_metric || "Score")}</span><strong>${formatNumber(best.rank_score ?? "n/a")}</strong></div>
      <div class="stat"><span>Latest delta</span><strong class="experiment-delta ${scoreDeltaClass(latest.score_delta)}">${formatScoreDelta(latest.score_delta)}</strong></div>
      <div class="stat"><span>Improved runs</span><strong>${formatNumber(summary.improved_runs ?? 0)}</strong></div>
    </div>
    <div class="experiment-grid">
      ${recent
        .map((model, index) => {
          const firstAction = model.next_actions?.[0] || "No recommendation saved for this run.";
          const trainedLabel =
            model.trained_models && model.candidate_models
              ? `${formatNumber(model.trained_models)}/${formatNumber(model.candidate_models)} trained`
              : "Older run";
          const featureLabel =
            model.model_feature_count && model.raw_feature_count
              ? `${formatNumber(model.model_feature_count)}/${formatNumber(model.raw_feature_count)} features`
              : "Feature detail unavailable";
          const topModels = (model.leaderboard_snapshot || [])
            .slice(0, 3)
            .map(
              (item, itemIndex) => `
                <li>
                  <span>#${itemIndex + 1} ${escapeHtml(item.model_name || "Model")}</span>
                  <strong>${formatNumber(item.rank_score ?? "n/a")}</strong>
                </li>
              `
            )
            .join("");

          return `
            <article class="experiment-card ${model.is_latest || index === 0 ? "latest" : ""} ${model.is_best_for_group ? "best" : ""}">
              <div class="experiment-card-top">
                <span>${model.is_latest || index === 0 ? "Latest" : `Run ${formatNumber(model.group_run_number || index + 1)}`}</span>
                <b>${model.is_best_for_group ? "Best" : escapeHtml(model.quality_label || "saved")}</b>
              </div>
              <strong>${escapeHtml(model.best_model || "Model")}</strong>
              <p>${escapeHtml(model.target_column || "target")} &middot; ${escapeHtml(model.problem_type || "problem")}</p>
              <div class="experiment-score">
                <span>${escapeHtml(model.primary_metric || "score")}</span>
                <strong>${formatNumber(model.rank_score ?? "n/a")}</strong>
              </div>
              <div class="experiment-meta">
                <span class="experiment-delta ${scoreDeltaClass(model.score_delta)}">${formatScoreDelta(model.score_delta)}</span>
                <span>${escapeHtml(trainedLabel)}</span>
                <span>${escapeHtml(featureLabel)}</span>
              </div>
              <p class="muted">${escapeHtml(firstAction)}</p>
              <details class="experiment-detail">
                <summary>Review run</summary>
                <div class="metric-list">${renderMetricBadges(model.holdout_metrics || {})}</div>
                ${topModels ? `<ol class="experiment-top-models">${topModels}</ol>` : ""}
                <code>${escapeHtml(model.prediction_api || "No API saved")}</code>
              </details>
            </article>
          `;
        })
        .join("")}
    </div>
    <div class="table-wrap experiment-table">
      <table>
        <thead>
          <tr><th>Run</th><th>Model</th><th>Target</th><th>Metric</th><th>Score</th><th>Delta</th><th>Trained</th><th>Features</th><th>Created</th></tr>
        </thead>
        <tbody>
          ${experiments
            .map(
              (model) => `
                <tr>
                  <td>${model.is_latest ? "Latest" : `Run ${formatNumber(model.group_run_number || "")}`}</td>
                  <td>${escapeHtml(model.best_model || "Model")}${model.is_best_for_group ? ' <span class="metric-pill">best</span>' : ""}</td>
                  <td>${escapeHtml(model.target_column || "")}</td>
                  <td>${escapeHtml(model.primary_metric || "")}</td>
                  <td>${formatNumber(model.rank_score ?? "n/a")}</td>
                  <td><span class="experiment-delta ${scoreDeltaClass(model.score_delta)}">${formatScoreDelta(model.score_delta)}</span></td>
                  <td>${formatNumber(model.trained_models ?? "")}/${formatNumber(model.candidate_models ?? "")}</td>
                  <td>${formatNumber(model.model_feature_count ?? "")}/${formatNumber(model.raw_feature_count ?? "")}</td>
                  <td>${escapeHtml(model.created_at || "")}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
  attachExperimentFilterEvents(container);
}

function attachExperimentFilterEvents(container) {
  container.querySelectorAll("[data-experiment-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.disabled) {
        return;
      }
      experimentFilterMode = button.dataset.experimentFilter;
      loadModelRegistry();
    });
  });
}

function renderModelRegistry(models, experimentHistory = null) {
  const container = document.querySelector("#modelRegistry");
  const preview = document.querySelector("#modelRegistryPreview");
  if (!models.length) {
    [container, preview].forEach((target) => {
      if (!target) {
        return;
      }
      target.className = "table-wrap empty-state";
      target.textContent = "Train a model to add it to the registry.";
    });
    renderRegistryStats(models);
    renderExperimentComparison(experimentHistory || []);
    renderExportCenter(models);
    renderPredictionPlaygroundSelector(models);
    return;
  }

  if (container) {
    container.className = "table-wrap";
    container.innerHTML = modelRegistryTable(models);
  }
  if (preview) {
    preview.className = "table-wrap";
    preview.innerHTML = modelRegistryTable(models, 3);
  }
  renderRegistryStats(models);
  renderExperimentComparison(experimentHistory || models);
  renderExportCenter(models);
  renderPredictionPlaygroundSelector(models);
}

async function loadModelRegistry() {
  const [response, experimentResponse] = await Promise.all([
    fetch("/api/models"),
    fetch(`/api/experiments?${buildExperimentQuery()}`),
  ]);
  if (!response.ok) {
    return;
  }

  const body = await response.json();
  modelRegistry = body.models || [];
  const experimentHistory = experimentResponse.ok ? await experimentResponse.json() : null;
  renderModelRegistry(modelRegistry, experimentHistory);
  updateDeployCommandBar();
  renderSidebarPulse();
  renderStudioOverview();
}

function getSelectedTargetProfile() {
  const target = document.querySelector("#targetColumn").value;
  return currentProfile?.columns.find((column) => column.name === target);
}

function isIdLikeColumn(name) {
  const normalized = String(name || "").toLowerCase().replaceAll("-", "_").replaceAll(" ", "_");
  return normalized === "id" || normalized.endsWith("_id") || normalized.endsWith("id");
}

function getAvailableFeatureColumns() {
  const target = document.querySelector("#targetColumn")?.value;
  return (currentProfile?.columns || []).filter((column) => column.name !== target);
}

function getSelectedFeatureColumns() {
  return getAvailableFeatureColumns()
    .filter((column) => selectedFeatureColumns.has(column.name))
    .map((column) => column.name);
}

function initializeFeatureSelection(force = false) {
  if (!currentProfile?.columns?.length) {
    return;
  }

  const available = getAvailableFeatureColumns();
  const availableNames = new Set(available.map((column) => column.name));
  const stillValid = [...selectedFeatureColumns].filter((name) => availableNames.has(name));

  if (force || !featureSelectionInitialized || !stillValid.length) {
    selectedFeatureColumns = new Set(available.map((column) => column.name));
    featureSelectionInitialized = true;
    return;
  }

  selectedFeatureColumns = new Set(stillValid);
}

function renderFeatureSelector() {
  const list = document.querySelector("#featureColumnList");
  const summary = document.querySelector("#featureSelectionSummary");
  if (!list || !summary) {
    return;
  }

  if (!currentProfile?.columns?.length) {
    list.className = "feature-list empty-state";
    list.textContent = "Feature columns will appear here.";
    summary.textContent = "Load a dataset to choose training columns.";
    return;
  }

  initializeFeatureSelection();
  const columns = getAvailableFeatureColumns();
  const selectedCount = getSelectedFeatureColumns().length;
  summary.textContent = `${selectedCount} of ${columns.length} columns selected for training. Target is excluded automatically.`;

  list.className = "feature-list";
  list.innerHTML = columns
    .map((column) => {
      const checked = selectedFeatureColumns.has(column.name);
      const hint = isIdLikeColumn(column.name) ? "ID-like" : column.type;
      return `
        <label class="feature-option ${checked ? "selected" : ""}">
          <input type="checkbox" data-feature-column="${escapeHtml(column.name)}" ${checked ? "checked" : ""} />
          <span>
            <strong>${escapeHtml(column.name)}</strong>
            <small>${escapeHtml(hint)} &middot; ${formatNumber(column.unique)} unique &middot; ${formatNumber(column.missing)} missing</small>
          </span>
        </label>
      `;
    })
    .join("");

  document.querySelectorAll("[data-feature-column]").forEach((checkbox) => {
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        selectedFeatureColumns.add(checkbox.dataset.featureColumn);
      } else {
        selectedFeatureColumns.delete(checkbox.dataset.featureColumn);
      }
      renderFeatureSelector();
      loadModelSuggestions();
    });
  });
}

function selectAllFeatures() {
  selectedFeatureColumns = new Set(getAvailableFeatureColumns().map((column) => column.name));
  featureSelectionInitialized = true;
  renderFeatureSelector();
  loadModelSuggestions();
}

function removeIdLikeFeatures() {
  selectedFeatureColumns = new Set(
    getAvailableFeatureColumns()
      .filter((column) => !isIdLikeColumn(column.name))
      .map((column) => column.name)
  );
  featureSelectionInitialized = true;
  renderFeatureSelector();
  loadModelSuggestions();
}

function updateProblemTypeSuggestion() {
  const target = document.querySelector("#targetColumn").value;
  if (!target) {
    return;
  }

  const problemType = document.querySelector("#problemType");
  problemType.value = getSuggestedProblemType(target);
  updateActiveTargetNote();
  renderWorkspaceOverview();
  initializeFeatureSelection();
  renderFeatureSelector();

  loadModelSuggestions();
}

async function loadModelSuggestions() {
  if (!currentDatasetId || !document.querySelector("#targetColumn").value) {
    document.querySelector("#modelSuggestions").className = "model-suggestions empty-state";
    document.querySelector("#modelSuggestions").textContent = "Load a dataset to get model recommendations.";
    return;
  }

  const container = document.querySelector("#modelSuggestions");
  container.className = "model-suggestions";
  container.innerHTML = "<p class=\"muted\">Finding best model candidates...</p>";

  const response = await fetch("/api/model-suggestions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: currentDatasetId,
      target_column: document.querySelector("#targetColumn").value,
      problem_type: document.querySelector("#problemType").value,
      feature_columns: getSelectedFeatureColumns(),
    }),
  });

  if (!response.ok) {
    container.className = "model-suggestions empty-state";
    container.textContent = await readErrorMessage(response);
    return;
  }

  const result = await response.json();
  const primaryMetricLabel = document.querySelector("#primaryMetricLabel");
  if (primaryMetricLabel) {
    primaryMetricLabel.textContent = result.primary_metric;
  }
  container.innerHTML = `
    <div class="suggestion-header">
      <div>
        <span class="panel-kicker">Recommended training plan</span>
        <h2>${escapeHtml(result.problem_type)} models for ${escapeHtml(result.target_column)}</h2>
      </div>
      <span class="metric-pill">${formatNumber(result.candidate_count)} candidates</span>
      <span class="metric-pill">Metric: ${escapeHtml(result.primary_metric)}</span>
    </div>
    <div class="dataset-mini">
      <span>${formatNumber(result.dataset_summary.rows_with_target)} rows</span>
      <span>${formatNumber(result.dataset_summary.feature_count)} raw features</span>
      <span>${formatNumber(result.dataset_summary.model_feature_count)} model features</span>
      <span>${formatNumber(result.dataset_summary.numeric_features)} numeric</span>
      <span>${formatNumber(result.dataset_summary.categorical_features)} categorical</span>
      <span>${formatNumber(result.dataset_summary.engineered_features)} engineered</span>
      <span>${formatNumber(result.dataset_summary.dropped_features)} dropped</span>
      <span>${formatNumber(result.dataset_summary.leakage_features || 0)} leakage guarded</span>
      <span>${formatNumber(result.dataset_summary.excluded_features)} excluded</span>
    </div>
    ${renderLeakageGuard(result.feature_plan?.leakage_columns || [])}
    <div class="model-card-grid">
      ${result.suggestions
        .map((model, index) => {
          const icons = ["LR", "RC", "SVM", "KN", "DT", "RF", "ET", "GB", "AB", "SVR"];
          return `
            <article class="model-card ${index === 0 ? "selected" : ""}">
              <span class="model-icon">${icons[index % icons.length]}</span>
              <div class="model-card-top">
                <strong>${escapeHtml(model.name)}</strong>
                <span>${escapeHtml(model.fit)}</span>
              </div>
              <p>${escapeHtml(model.why)}</p>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

async function uploadCsv(event) {
  event.preventDefault();
  const file = csvFileInput.files[0];
  if (!file) {
    alert("Choose a CSV file first.");
    return;
  }

  await uploadSelectedFile(file);
}

async function uploadSelectedFile(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    alert("Please upload a CSV file.");
    return;
  }

  selectedFileName.textContent = file.name;
  setStatus("Uploading CSV");
  setBusy(true);
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/upload-csv", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Upload failed");
    setBusy(false);
    return;
  }

  const result = await response.json();
  renderProfile(result);
  showToast("Upload complete", `${result.filename || file.name} profile is ready.`);
  setBusy(false);
}

function setDroppedFile(file) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  csvFileInput.files = transfer.files;
  selectedFileName.textContent = file.name;
}

function preventFileOpen(event) {
  event.preventDefault();
  event.stopPropagation();
  if (event.dataTransfer) {
    event.dataTransfer.dropEffect = "copy";
  }
}

const uploadDropTargets = [dropZone, uploadFormElement].filter(Boolean);

uploadDropTargets.forEach((target) => {
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    target.addEventListener(eventName, preventFileOpen);
  });
});

uploadDropTargets.forEach((target) => {
  ["dragenter", "dragover"].forEach((eventName) => {
    target.addEventListener(eventName, (event) => {
      dropZone.classList.add("drag-over");
      const dragTypes = Array.from(event.dataTransfer?.types || []);
      const hasBucketItem = dragTypes.includes("application/x-automate-sample");
      setStatus(hasBucketItem ? "Drop sample into upload bucket" : "Drop CSV to upload");
    });
  });
});

uploadDropTargets.forEach((target) => {
  ["dragleave", "drop"].forEach((eventName) => {
    target.addEventListener(eventName, () => {
      dropZone.classList.remove("drag-over");
    });
  });
});

async function handleCsvDrop(event) {
  const sampleId = event.dataTransfer.getData("application/x-automate-sample");
  if (sampleId) {
    dropZone.classList.remove("drag-over", "bucket-ready");
    await loadSampleDataset(sampleId);
    return;
  }

  const textPayload = event.dataTransfer.getData("text/plain") || "";
  if (textPayload.startsWith("sample:")) {
    dropZone.classList.remove("drag-over", "bucket-ready");
    await loadSampleDataset(textPayload.replace("sample:", ""));
    return;
  }

  const file = event.dataTransfer.files[0];
  if (!file) {
    return;
  }

  dropZone.classList.remove("drag-over", "bucket-ready");
  setDroppedFile(file);
  await uploadSelectedFile(file);
}

dropZone.addEventListener("drop", handleCsvDrop);
uploadFormElement.addEventListener("drop", handleCsvDrop);

function syncSelectedFileState(file) {
  selectedFileName.textContent = file ? file.name : "No file selected";
  dropZone.classList.toggle("has-file", Boolean(file));
  if (file) {
    setStatus("CSV ready to upload");
  }
}

csvFileInput.addEventListener("change", () => {
  const file = csvFileInput.files[0];
  syncSelectedFileState(file);
});

async function connectSqlite(event) {
  event.preventDefault();
  const dbPath = document.querySelector("#dbPath").value;
  const tableName = document.querySelector("#tableName").value;

  const response = await fetch("/api/connect-sqlite", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ db_path: dbPath, table_name: tableName }),
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    return;
  }

  renderProfile(await response.json());
}

function renderCharts(charts) {
  if (!charts.length) {
    return "<p class=\"muted\">No chart suggestions available for this dataset.</p>";
  }

  return charts
    .map((chart) => {
      const values = chart.y || chart.values || [];
      const labels = chart.x || values.map((_, index) => `Row ${index + 1}`);
      const max = Math.max(...values.map((value) => Number(value) || 0), 1);
      let visual = "";

      if (chart.type === "pie") {
        visual = renderPieChart(labels, values);
      } else if (chart.type === "scatter") {
        visual = renderScatterChart(chart);
      } else if (chart.type === "heatmap") {
        visual = renderHeatmap(chart);
      } else if (chart.type === "line") {
        visual = renderLineChart(labels, values);
      } else {
        visual = renderBarLikeChart(labels, values, max);
      }

      return `
        <article class="chart-card">
          <h2>${escapeHtml(chart.title)}</h2>
          <p class="muted">${escapeHtml(chart.type)}. ${escapeHtml(chart.reason)}</p>
          ${visual}
        </article>
      `;
    })
    .join("");
}

function renderBarLikeChart(labels, values, max) {
      const bars = values
        .slice(0, 8)
        .map((value, index) => {
          const width = Math.max(4, ((Number(value) || 0) / max) * 100);
          return `
            <div class="bar-row">
              <span>${escapeHtml(String(labels[index]).slice(0, 18))}</span>
              <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
              <span>${Number(value).toFixed(0)}</span>
            </div>
          `;
        })
        .join("");

  return `<div class="chart-bars">${bars}</div>`;
}

function renderPieChart(labels, values) {
  const colors = ["#0f766e", "#0369a1", "#d97706", "#7c3aed", "#db2777", "#475569"];
  const total = values.reduce((sum, value) => sum + Number(value || 0), 0) || 1;
  let start = 0;
  const segments = values.map((value, index) => {
    const size = (Number(value || 0) / total) * 100;
    const segment = `${colors[index % colors.length]} ${start}% ${start + size}%`;
    start += size;
    return segment;
  });

  const legend = values
    .map(
      (value, index) => `
        <span><i style="background:${colors[index % colors.length]}"></i>${escapeHtml(labels[index])} (${formatNumber(Number(value) || 0)})</span>
      `
    )
    .join("");

  return `
    <div class="pie-wrap">
      <div class="pie" style="background: conic-gradient(${segments.join(", ")});"></div>
      <div class="legend">${legend}</div>
    </div>
  `;
}

function renderLineChart(labels, values) {
  const numericValues = values.map((value) => Number(value) || 0);
  const max = Math.max(...numericValues, 1);
  const min = Math.min(...numericValues, 0);
  const range = max - min || 1;
  const points = numericValues
    .map((value, index) => {
      const x = 20 + (index / Math.max(numericValues.length - 1, 1)) * 300;
      const y = 150 - ((value - min) / range) * 120;
      return `${x},${y}`;
    })
    .join(" ");

  return `
    <svg class="mini-chart" viewBox="0 0 340 170" role="img">
      <polyline points="${points}" fill="none" stroke="#0369a1" stroke-width="4" />
      <line x1="20" y1="150" x2="320" y2="150" stroke="#cbd5e1" />
      <line x1="20" y1="20" x2="20" y2="150" stroke="#cbd5e1" />
    </svg>
  `;
}

function renderScatterChart(chart) {
  const points = chart.points || [];
  const xValues = points.map((point) => Number(point[chart.x_label]) || 0);
  const yValues = points.map((point) => Number(point[chart.y_label]) || 0);
  const xMin = Math.min(...xValues, 0);
  const xMax = Math.max(...xValues, 1);
  const yMin = Math.min(...yValues, 0);
  const yMax = Math.max(...yValues, 1);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const dots = points
    .slice(0, 120)
    .map((point) => {
      const x = 24 + ((Number(point[chart.x_label]) - xMin) / xRange) * 292;
      const y = 146 - ((Number(point[chart.y_label]) - yMin) / yRange) * 118;
      return `<circle cx="${x}" cy="${y}" r="4" fill="#0f766e" opacity="0.78" />`;
    })
    .join("");

  return `
    <svg class="mini-chart" viewBox="0 0 340 170" role="img">
      <line x1="20" y1="150" x2="320" y2="150" stroke="#cbd5e1" />
      <line x1="20" y1="20" x2="20" y2="150" stroke="#cbd5e1" />
      ${dots}
    </svg>
  `;
}

function renderHeatmap(chart) {
  const cells = chart.matrix
    .map((row, rowIndex) =>
      row
        .map((value, colIndex) => {
          const intensity = Math.min(Math.abs(Number(value)), 1);
          const color = value >= 0 ? `rgba(15, 118, 110, ${0.15 + intensity * 0.75})` : `rgba(217, 119, 6, ${0.15 + intensity * 0.75})`;
      return `<td style="background:${color}">${Number(value).toFixed(2)}</td>`;
        })
        .join("")
    )
    .map((cells, index) => `<tr><th>${escapeHtml(chart.columns[index])}</th>${cells}</tr>`)
    .join("");

  return `
    <div class="table-wrap">
      <table class="heatmap">
        <thead><tr><th></th>${chart.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
        <tbody>${cells}</tbody>
      </table>
    </div>
  `;
}

function dashboardRequestUrl(datasetId) {
  const url = new URL(`/api/dashboard/${datasetId}`, window.location.origin);
  if (Object.keys(dashboardFilters).length) {
    url.searchParams.set("filters", JSON.stringify(dashboardFilters));
  }
  return url;
}

function renderDashboardFilters(filters, activeFilters = {}) {
  const container = document.querySelector("#filterList");
  if (!filters?.length) {
    container.className = "filter-list empty-state";
    container.textContent = "No filter suggestions found.";
    return;
  }

  container.className = "filter-list filter-controls";
  container.innerHTML = filters
    .map(
      (filter) => `
        <label class="filter-control">
          <span>${escapeHtml(filter.column)}</span>
          <select data-dashboard-filter="${escapeHtml(filter.column)}">
            <option value="">All</option>
            ${filter.options
              .map(
                (option) => `
                  <option value="${escapeHtml(option.value)}" ${String(activeFilters[filter.column] ?? "") === String(option.value) ? "selected" : ""}>
                    ${escapeHtml(option.value)} (${formatNumber(option.count)})
                  </option>
                `
              )
              .join("")}
          </select>
        </label>
      `
    )
    .join("");

  document.querySelectorAll("[data-dashboard-filter]").forEach((select) => {
    select.addEventListener("change", async () => {
      const column = select.dataset.dashboardFilter;
      if (select.value) {
        dashboardFilters[column] = select.value;
      } else {
        delete dashboardFilters[column];
      }
      await loadDashboard();
    });
  });
}

async function loadDashboard() {
  if (!currentDatasetId) {
    alert("Upload data first.");
    return;
  }

  setStatus("Generating dashboard");
  setBusy(true);
  const response = await fetch(dashboardRequestUrl(currentDatasetId));
  if (!response.ok) {
    alert(await readErrorMessage(response));
    setBusy(false);
    return;
  }

  const data = await response.json();
  lastDashboardData = data;
  const currentDataset = datasetRegistry.find((dataset) => dataset.dataset_id === currentDatasetId);
  document.querySelector("#dashboardTitle").textContent = currentDataset ? `${currentDataset.name} Dashboard` : "Analytics Dashboard";
  document.querySelector("#dashboardSubtitle").textContent = `${formatNumber(data.active_row_count)} active rows from ${formatNumber(data.total_row_count)} total rows`;
  document.querySelector("#kpiGrid").innerHTML = data.kpis
    .map((kpi) => `<div class="kpi">${escapeHtml(kpi.label)}<strong>${formatNumber(kpi.value)}</strong></div>`)
    .join("");

  document.querySelector("#summaryList").innerHTML = data.summaries
    .map((summary) => `<div>${escapeHtml(summary)}</div>`)
    .join("");

  document.querySelector("#chartList").innerHTML = renderCharts(data.charts);

  renderDashboardFilters(data.available_filters, data.active_filters);
  renderStudioOverview();
  renderSidebarPulse();

  setStatus("Dashboard ready");
  setBusy(false);
  showView("dashboard");
}

async function quickDashboard() {
  if (!currentDatasetId && datasetRegistry.length) {
    await selectDataset(datasetRegistry[0].dataset_id);
  }
  if (currentDatasetId) {
    await loadDashboard();
  } else {
    showView("dashboard");
  }
}

async function quickTrain() {
  if (!currentDatasetId && datasetRegistry.length) {
    await selectDataset(datasetRegistry[0].dataset_id);
  }
  showView("model");
  document.querySelector("#targetColumn")?.focus();
}

function renderMetricBadges(metrics) {
  return Object.entries(metrics || {})
    .map(([key, value]) => `<span class="metric-pill">${escapeHtml(key)}: ${formatNumber(value)}</span>`)
    .join("");
}

function renderExplainabilityStudio(studio, fallbackFeatures = []) {
  const features = studio?.features?.length ? studio.features : fallbackFeatures || [];

  if (!studio?.available && !features.length) {
    return "<p class=\"muted\">Explainability could not be calculated for this model.</p>";
  }

  const summary = studio?.summary || "Permutation importance shows how much each feature changes validation score.";
  const cards = studio?.cards || [];
  const actions = studio?.actions || [];
  const maxShare = Math.max(...features.map((item) => Math.abs(item.share || item.importance || 0)), 0.01);

  return `
    <div class="explainability-studio">
      <div class="explainability-head">
        <div>
          <span class="panel-kicker">Advanced Explainability</span>
          <strong>${escapeHtml(studio?.method || "permutation_importance")}</strong>
        </div>
        <span class="metric-pill">${features.length} drivers</span>
      </div>
      <p class="muted">${escapeHtml(summary)}</p>
      ${
        cards.length
          ? `<div class="explainability-card-grid">
              ${cards
                .map(
                  (card) => `
                    <div class="explainability-tile">
                      <span>${escapeHtml(card.label)}</span>
                      <strong>${formatNumber(card.value)}</strong>
                      <small>${escapeHtml(card.note || "")}</small>
                    </div>
                  `
                )
                .join("")}
            </div>`
          : ""
      }
      <div class="explainability-feature-list">
        ${features
          .map((item) => {
            const share = Math.abs(item.share || item.importance || 0);
            const width = Math.max(8, Math.min((share / maxShare) * 100, 100));
            const impactKey = item.impact_key || (item.importance >= 0 ? "helpful" : "weak_signal");
            return `
              <div class="explainability-feature ${impactKey}">
                <div>
                  <strong>${escapeHtml(item.feature)}</strong>
                  <span>${escapeHtml(item.strength || "Signal")} &middot; ${escapeHtml(item.impact || "Impact")}</span>
                </div>
                <b>${formatNumber(item.importance)}</b>
                <div class="importance-track"><div style="width:${width}%"></div></div>
                <small>${escapeHtml(item.interpretation || "")}</small>
              </div>
            `;
          })
          .join("")}
      </div>
      ${
        actions.length
          ? `<ul class="explainability-actions">
              ${actions.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>`
          : ""
      }
    </div>
  `;
}

function renderTargetSummary(summary) {
  if (!summary) {
    return "";
  }

  if (summary.kind === "classification") {
    return `
      <div class="target-balance">
        ${summary.classes
          .map(
            (item) => `
              <div>
                <strong>${escapeHtml(item.label)}</strong>
                <span>${formatNumber(item.count)} rows</span>
                <div class="importance-track"><div style="width:${Math.max(4, item.share * 100)}%"></div></div>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  return `
    <div class="metric-list">
      <span>Mean ${formatNumber(summary.mean)}</span>
      <span>Median ${formatNumber(summary.median)}</span>
      <span>Min ${formatNumber(summary.min)}</span>
      <span>Max ${formatNumber(summary.max)}</span>
      <span>Std ${formatNumber(summary.std)}</span>
    </div>
  `;
}

function renderActionPlan(actions) {
  if (!actions?.length) {
    return "";
  }

  return `
    <div class="action-plan">
      <strong>Recommended next steps</strong>
      ${actions.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
    </div>
  `;
}

function renderLeakageGuard(leakageColumns = []) {
  if (!leakageColumns?.length) {
    return "";
  }

  return `
    <section class="leakage-guard">
      <div class="leakage-head">
        <div>
          <span class="panel-kicker">Data leakage guard</span>
          <strong>${formatNumber(leakageColumns.length)} suspicious feature(s) auto-excluded</strong>
          <small>These columns look too close to the target and can create fake accuracy.</small>
        </div>
        <span class="metric-pill">Protected training</span>
      </div>
      <div class="leakage-list">
        ${leakageColumns
          .map(
            (item) => `
              <article class="leakage-item ${escapeHtml(item.severity || "")}">
                <span>${escapeHtml(item.severity || "warning")}</span>
                <strong>${escapeHtml(item.name)}</strong>
                <small>${escapeHtml(item.reason)}</small>
                <em>${escapeHtml(item.action || "Excluded from training.")}</em>
              </article>
            `
          )
          .join("")}
      </div>
    </section>
  `;
}


function renderTuningStudio(tuning, primaryMetric) {
  if (!tuning?.available) {
    return "";
  }

  const results = tuning.results || [];
  const savedLabel = tuning.saved_tuned_model ? `Saved tuned winner: ${tuning.best_tuned_model}` : "Base winner kept";
  return `
    <section class="tuning-studio ${tuning.saved_tuned_model ? "saved" : ""}">
      <div class="tuning-head">
        <div>
          <span class="panel-kicker">Hyperparameter tuning</span>
          <strong>${escapeHtml(savedLabel)}</strong>
          <small>Tuned ${formatNumber(tuning.tuned_count || 0)} model(s); ${formatNumber(tuning.improved_count || 0)} improved on ${escapeHtml(primaryMetric)}.</small>
        </div>
        <span class="metric-pill">Top ${formatNumber((tuning.selected_models || []).length)} searched</span>
      </div>
      <div class="tuning-result-grid">
        ${results
          .map((item) => {
            const params = item.best_params
              ? Object.entries(item.best_params)
                  .map(([key, value]) => `${escapeHtml(key)}=${escapeHtml(value)}`)
                  .join(", ")
              : escapeHtml(item.reason || "No parameter search run");
            return `
              <article class="tuning-result ${escapeHtml(item.status || "")}">
                <span>${escapeHtml((item.status || "checked").replaceAll("_", " "))}</span>
                <strong>${escapeHtml(item.model_name)}</strong>
                <div class="tuning-score-row">
                  <small>Base ${formatNumber(item.base_score ?? "n/a")}</small>
                  <small>Tuned ${formatNumber(item.tuned_score ?? "n/a")}</small>
                  <small>Gain ${formatNumber(item.improvement ?? "n/a")}</small>
                </div>
                <em>${params}</em>
                <small>${formatNumber(item.trials || 0)} trial(s)</small>
              </article>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}


function renderAccuracyBooster(booster) {
  if (!booster?.available) {
    return "";
  }

  const comparison = booster.comparison || {};
  const cleaning = booster.cleaning || {};
  const strategy = booster.strategy || [];
  const deltaClass = scoreDeltaClass(comparison.delta);
  const changedSteps = (cleaning.steps || []).filter((step) => step.changed);

  return `
    <section class="accuracy-booster ${escapeHtml(booster.status || "")}">
      <div class="accuracy-booster-head">
        <div>
          <span class="panel-kicker">Auto Accuracy Booster</span>
          <strong>${escapeHtml(booster.summary || "Boosted training completed.")}</strong>
          <small>${escapeHtml(comparison.metric || "score")} comparison against latest saved run</small>
        </div>
        <span class="booster-delta ${deltaClass}">${formatScoreDelta(comparison.delta)}</span>
      </div>
      <div class="booster-comparison-grid">
        <div class="stat">
          <span>Previous model</span>
          <strong>${escapeHtml(comparison.previous_model || "No baseline")}</strong>
          <small>${formatNumber(comparison.previous_score ?? "n/a")}</small>
        </div>
        <div class="stat">
          <span>Boosted model</span>
          <strong>${escapeHtml(comparison.boosted_model || "Model")}</strong>
          <small>${formatNumber(comparison.boosted_score ?? "n/a")}</small>
        </div>
        <div class="stat">
          <span>Cleaning score</span>
          <strong>${formatNumber(cleaning.before_score ?? "n/a")} -> ${formatNumber(cleaning.after_score ?? "n/a")}</strong>
          <small>${formatNumber(changedSteps.length)} action(s) applied</small>
        </div>
      </div>
      <div class="booster-strategy-grid">
        ${strategy
          .map(
            (item) => `
              <div class="booster-strategy ${escapeHtml(item.status || "")}">
                <span>${escapeHtml(item.status || "checked")}</span>
                <strong>${escapeHtml(item.label)}</strong>
                <small>${escapeHtml(item.detail)}</small>
              </div>
            `
          )
          .join("")}
      </div>
      ${
        changedSteps.length
          ? `<div class="booster-cleaning-list">
              ${changedSteps
                .map((step) => `<span>${escapeHtml(step.title)}: ${escapeHtml(step.impact || "applied")}</span>`)
                .join("")}
            </div>`
          : ""
      }
    </section>
  `;
}


function renderAccuracyImprover(plan) {
  if (!plan?.available) {
    return "";
  }

  const checks = plan.checks || [];
  const actions = plan.actions || [];
  return `
    <section class="accuracy-improver ${escapeHtml(plan.status || "")}">
      <div class="accuracy-improver-head">
        <div>
          <span class="panel-kicker">Accuracy improver</span>
          <strong>${escapeHtml(plan.best_model || "Best model")}</strong>
          <small>${escapeHtml(plan.summary || "Training quality analysis is ready.")}</small>
        </div>
        <span class="improver-status">${escapeHtml((plan.status || "ready").replaceAll("_", " "))}</span>
      </div>
      <div class="improver-check-grid">
        ${checks
          .map(
            (check) => `
              <div class="improver-check ${escapeHtml(check.status || "")}">
                <span>${escapeHtml(check.status || "check")}</span>
                <strong>${escapeHtml(check.label)}</strong>
                <small>${escapeHtml(check.detail)}</small>
              </div>
            `
          )
          .join("")}
      </div>
      ${
        actions.length
          ? `<div class="improver-actions"><strong>Next fixes</strong>${actions.map((action) => `<span>${escapeHtml(action)}</span>`).join("")}</div>`
          : ""
      }
    </section>
  `;
}


function renderDiagnostics(diagnostics) {
  const container = document.querySelector("#modelDiagnostics");
  if (!diagnostics) {
    container.className = "diagnostic-grid empty-state";
    container.textContent = "Diagnostics will appear after training.";
    return;
  }

  container.className = "diagnostic-grid";
  if (diagnostics.kind === "classification") {
    const summary = diagnostics.summary || {};
    const topConfusion = summary.top_confusion
      ? `${summary.top_confusion.actual} -> ${summary.top_confusion.predicted} (${formatNumber(summary.top_confusion.count)} rows)`
      : "No repeated confusion";
    const weakestClass = summary.weakest_class
      ? `${summary.weakest_class.label} recall ${formatNumber(summary.weakest_class.recall)}`
      : "No weak class detected";
    const header = diagnostics.labels.map((label) => `<th>Pred ${escapeHtml(label)}</th>`).join("");
    const matrixRows = diagnostics.confusion_matrix
      .map(
        (row, index) => `
          <tr>
            <th>Actual ${escapeHtml(diagnostics.labels[index])}</th>
            ${row.map((value) => `<td>${formatNumber(value)}</td>`).join("")}
          </tr>
        `
      )
      .join("");
    const reportRows = diagnostics.class_report
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(row.label)}</td>
            <td>${formatNumber(row.precision)}</td>
            <td>${formatNumber(row.recall)}</td>
            <td>${formatNumber(row.f1)}</td>
            <td>${formatNumber(row.support)}</td>
          </tr>
        `
      )
      .join("");
    const mistakeRows = (diagnostics.mistake_samples || [])
      .map(
        (row) => `
          <tr>
            <td>${formatNumber(row.row)}</td>
            <td>${escapeHtml(row.actual)}</td>
            <td>${escapeHtml(row.predicted)}</td>
          </tr>
        `
      )
      .join("");
    const predictionDistribution = (diagnostics.prediction_distribution || [])
      .map((item) => `<span>${escapeHtml(item.label)}: ${formatNumber(item.count)}</span>`)
      .join("");

    container.innerHTML = `
      <div class="diagnostic-card">
        <h3>Quality snapshot</h3>
        <div class="metric-list block">
          <span>Accuracy ${formatNumber(summary.accuracy)}</span>
          <span>Error rate ${formatNumber(summary.error_rate)}</span>
          <span>Errors ${formatNumber(summary.errors)}</span>
          <span>Holdout ${formatNumber(summary.holdout_rows)}</span>
        </div>
        <p class="muted">Top confusion: ${escapeHtml(topConfusion)}</p>
        <p class="muted">Weakest class: ${escapeHtml(weakestClass)}</p>
        <div class="metric-list">${predictionDistribution}</div>
      </div>
      <div class="table-wrap diagnostic-card">
        <table class="matrix-table">
          <thead><tr><th></th>${header}</tr></thead>
          <tbody>${matrixRows}</tbody>
        </table>
      </div>
      <div class="table-wrap diagnostic-card">
        <table>
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>${reportRows}</tbody>
        </table>
      </div>
      <div class="table-wrap diagnostic-card">
        <h3>Misclassified holdout rows</h3>
        ${
          mistakeRows
            ? `<table><thead><tr><th>Row</th><th>Actual</th><th>Predicted</th></tr></thead><tbody>${mistakeRows}</tbody></table>`
            : "<p class=\"muted\">No holdout mistakes found for the best model.</p>"
        }
      </div>
    `;
    return;
  }

  const residuals = diagnostics.residual_summary || {};
  const sampleRows = diagnostics.samples
    .map(
      (row) => `
        <tr>
          <td>${formatNumber(row.row)}</td>
          <td>${formatNumber(row.actual)}</td>
          <td>${formatNumber(row.predicted)}</td>
          <td>${formatNumber(row.residual)}</td>
          <td>${formatNumber(row.abs_error)}</td>
        </tr>
      `
    )
    .join("");
  container.innerHTML = `
    <div class="diagnostic-card">
      <h3>Quality snapshot</h3>
      <div class="metric-list block">
        <span>R2 ${formatNumber(residuals.r2)}</span>
        <span>MAE ${formatNumber(residuals.mae)}</span>
        <span>RMSE ${formatNumber(residuals.rmse)}</span>
        <span>MAPE ${formatNumber(residuals.mape)}</span>
        <span>Holdout ${formatNumber(residuals.holdout_rows)}</span>
      </div>
      <span>Mean error ${formatNumber(residuals.mean_error)}</span>
      <span>Median error ${formatNumber(residuals.median_error)}</span>
      <span>Largest over ${formatNumber(residuals.largest_over_prediction)}</span>
      <span>Largest under ${formatNumber(residuals.largest_under_prediction)}</span>
    </div>
    <div class="table-wrap diagnostic-card">
      <h3>Largest residuals</h3>
      <table>
        <thead><tr><th>Row</th><th>Actual</th><th>Predicted</th><th>Residual</th><th>Abs error</th></tr></thead>
        <tbody>${sampleRows}</tbody>
      </table>
    </div>
  `;
}

function nearestThresholdMetrics(threshold) {
  if (!currentThresholdAnalysis?.curve?.length) {
    return null;
  }
  return currentThresholdAnalysis.curve.reduce((closest, item) =>
    Math.abs(item.threshold - threshold) < Math.abs(closest.threshold - threshold) ? item : closest
  );
}

function updateThresholdSelection(value) {
  const threshold = Number(value);
  currentThreshold = threshold;
  const valueLabel = document.querySelector("#thresholdValue");
  const metricBox = document.querySelector("#thresholdMetrics");
  const metrics = nearestThresholdMetrics(threshold);

  if (valueLabel) {
    valueLabel.textContent = threshold.toFixed(2);
  }
  document.querySelectorAll("[data-threshold-row]").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.thresholdRow) === threshold);
  });
  if (metricBox && metrics) {
    metricBox.innerHTML = `
      <span>Precision ${formatNumber(metrics.precision)}</span>
      <span>Recall ${formatNumber(metrics.recall)}</span>
      <span>F1 ${formatNumber(metrics.f1)}</span>
      <span>Accuracy ${formatNumber(metrics.accuracy)}</span>
      <span>Positive rate ${formatNumber(metrics.predicted_positive_rate)}</span>
    `;
  }
}

function renderThresholdTuning(analysis) {
  const panel = document.querySelector("#thresholdPanel");
  currentThreshold = null;
  currentThresholdAnalysis = analysis;

  if (!panel) {
    return;
  }

  if (!analysis?.available) {
    panel.className = "panel full threshold-panel empty-state";
    panel.textContent = analysis?.reason || "Threshold tuning will appear for binary classifiers with probability output.";
    return;
  }

  const recommendedThreshold = Number(analysis.recommended_threshold || 0.5);
  currentThreshold = recommendedThreshold;
  const rows = analysis.curve
    .map(
      (item) => `
        <tr data-threshold-row="${item.threshold}">
          <td>${formatNumber(item.threshold)}</td>
          <td>${formatNumber(item.precision)}</td>
          <td>${formatNumber(item.recall)}</td>
          <td>${formatNumber(item.f1)}</td>
          <td>${formatNumber(item.accuracy)}</td>
          <td>${formatNumber(item.predicted_positive)}</td>
        </tr>
      `
    )
    .join("");

  panel.className = "panel full threshold-panel";
  panel.innerHTML = `
    <div class="panel-heading">
      <div>
        <div class="panel-kicker">Threshold tuning</div>
        <h2>Precision and recall control</h2>
      </div>
      <span class="metric-pill">Positive class: ${escapeHtml(analysis.positive_class)}</span>
    </div>
    <div class="threshold-control">
      <label>
        <span>Decision threshold</span>
        <strong id="thresholdValue">${recommendedThreshold.toFixed(2)}</strong>
      </label>
      <input id="thresholdSlider" type="range" min="0.2" max="0.8" step="0.1" value="${recommendedThreshold}" />
      <div id="thresholdMetrics" class="metric-list block"></div>
    </div>
    <div class="table-wrap">
      <table class="threshold-table">
        <thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>Accuracy</th><th>Predicted positive</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;

  document.querySelector("#thresholdSlider").addEventListener("input", (event) => {
    updateThresholdSelection(event.target.value);
  });
  document.querySelectorAll("[data-threshold-row]").forEach((row) => {
    row.addEventListener("click", () => {
      document.querySelector("#thresholdSlider").value = row.dataset.thresholdRow;
      updateThresholdSelection(row.dataset.thresholdRow);
    });
  });
  updateThresholdSelection(recommendedThreshold);
}

function applyTrainingResult(result, options = {}) {
  currentModelId = result.model_id;
  currentPredictionApi = result.prediction_api;
  const primaryMetricLabel = document.querySelector("#primaryMetricLabel");
  if (primaryMetricLabel) {
    primaryMetricLabel.textContent = result.primary_metric;
  }

  document.querySelector("#trainingSummary").className = "train-summary";
  document.querySelector("#trainingSummary").innerHTML = `
    <div class="summary-row compact">
      <div class="stat"><span>Train rows</span><strong>${formatNumber(result.training_summary.train_rows)}</strong></div>
      <div class="stat"><span>Test rows</span><strong>${formatNumber(result.training_summary.test_rows)}</strong></div>
      <div class="stat"><span>Model features</span><strong>${formatNumber(result.training_summary.feature_count)}</strong></div>
      <div class="stat"><span>Raw features</span><strong>${formatNumber(result.training_summary.raw_feature_count)}</strong></div>
      <div class="stat"><span>Candidates</span><strong>${formatNumber(result.training_summary.candidate_models)}</strong></div>
      <div class="stat"><span>Trained</span><strong>${formatNumber(result.training_summary.trained_models)}</strong></div>
      <div class="stat"><span>Failed</span><strong>${formatNumber(result.training_summary.failed_models)}</strong></div>
      <div class="stat"><span>Engineered</span><strong>${formatNumber(result.training_summary.engineered_features?.length || 0)}</strong></div>
      <div class="stat"><span>Dropped</span><strong>${formatNumber(result.training_summary.dropped_features?.length || 0)}</strong></div>
      <div class="stat"><span>Leakage guard</span><strong>${formatNumber(result.training_summary.leakage_features?.length || 0)}</strong></div>
      <div class="stat"><span>Excluded</span><strong>${formatNumber(result.training_summary.excluded_feature_count || 0)}</strong></div>
      <div class="stat"><span>Balance</span><strong>${escapeHtml(result.training_summary.class_balance_severity || "n/a")}</strong></div>
      <div class="stat"><span>Balanced models</span><strong>${formatNumber(result.training_summary.balanced_candidate_count || 0)}</strong></div>
      <div class="stat"><span>Tuned</span><strong>${formatNumber(result.training_summary.tuned_models || 0)}</strong></div>
      <div class="stat"><span>Tune gains</span><strong>${formatNumber(result.training_summary.improved_tuned_models || 0)}</strong></div>
      <div class="stat wide"><span>Best model</span><strong>${escapeHtml(result.best_model)}</strong></div>
      <div class="stat"><span>Best ${escapeHtml(result.primary_metric)}</span><strong>${formatNumber(result.leaderboard?.[0]?.rank_score ?? "")}</strong></div>
    </div>
    ${renderTargetSummary(result.target_summary)}
  `;

  document.querySelector("#leaderboard").innerHTML = `
    <p class="metric-note">${escapeHtml(result.metric_note)}</p>
    ${renderAccuracyBooster(result.accuracy_booster)}
    ${
      result.quality_insights?.length
        ? `<div class="training-insights">${result.quality_insights.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`
        : ""
    }
    ${renderLeakageGuard(result.training_summary?.leakage_features || [])}
    ${renderTuningStudio(result.tuning_studio, result.primary_metric)}
    ${renderAccuracyImprover(result.accuracy_improver)}
    ${renderActionPlan(result.next_actions)}
    <div class="metric-list block">${renderMetricBadges(result.baseline_metrics)}</div>
    <table>
      <thead>
        <tr><th>Rank</th><th>Model</th><th>Status</th><th>${escapeHtml(result.primary_metric)}</th><th>CV</th><th>Holdout metrics</th></tr>
      </thead>
      <tbody>
        ${result.leaderboard
          .map(
            (item, index) => {
              const isFailed = item.status === "failed";
              return `
              <tr class="${isFailed ? "model-failed" : ""}">
                <td>${index + 1}</td>
                <td>${escapeHtml(item.model_name)}</td>
                <td><span class="metric-pill">${escapeHtml(isFailed ? "failed" : item.quality_label || item.status || "trained")}</span></td>
                <td>${isFailed ? "&mdash;" : formatNumber(item.rank_score)}</td>
                <td>${item.cross_validation.available ? `${formatNumber(item.cross_validation.mean)} +/- ${formatNumber(item.cross_validation.std)}` : escapeHtml(item.cross_validation.reason)}</td>
                <td>${isFailed ? escapeHtml(item.error || "Could not train on this dataset.") : `<div class="metric-list">${renderMetricBadges(item.metrics)}</div>`}</td>
              </tr>
            `;
            }
          )
          .join("")}
      </tbody>
    </table>
  `;

  document.querySelector("#explainability").innerHTML = renderExplainabilityStudio(
    result.explainability_studio,
    result.explainability || []
  );

  document.querySelector("#apiBox").textContent = `POST ${result.prediction_api}`;
  renderDiagnostics(result.diagnostics);
  renderThresholdTuning(result.threshold_analysis);
  document.querySelector("#predictPayload").value = JSON.stringify(
    {
      rows: [buildExamplePredictionRow()],
    },
    null,
    2
  );

  setStatus(options.statusText || `Best model: ${result.best_model}`);
  showToast(options.toastTitle || "Training complete", options.toastBody || `${result.best_model} is ranked best for ${result.target_column}.`);
  loadModelRegistry();
  showView("model");
}

async function trainModels(event) {
  event.preventDefault();
  if (!currentDatasetId) {
    alert("Upload data first.");
    return;
  }
  const featureColumns = getSelectedFeatureColumns();
  if (!featureColumns.length) {
    alert("Select at least one feature column before training.");
    return;
  }

  setStatus("Training models");
  setBusy(true);
  const response = await fetch("/api/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: currentDatasetId,
      target_column: document.querySelector("#targetColumn").value,
      problem_type: document.querySelector("#problemType").value,
      feature_columns: featureColumns,
    }),
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Training failed");
    setBusy(false);
    return;
  }

  const result = await response.json();
  applyTrainingResult(result);
  setBusy(false);
}

async function runAccuracyBooster() {
  if (!currentDatasetId) {
    alert("Upload data first.");
    return;
  }
  const featureColumns = getSelectedFeatureColumns();
  if (!featureColumns.length) {
    alert("Select at least one feature column before improving accuracy.");
    return;
  }

  setStatus("Improving accuracy");
  setBusy(true);
  const response = await fetch("/api/accuracy-booster", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_id: currentDatasetId,
      target_column: document.querySelector("#targetColumn").value,
      problem_type: document.querySelector("#problemType").value,
      feature_columns: featureColumns,
    }),
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Accuracy booster failed");
    setBusy(false);
    return;
  }

  const result = await response.json();
  const comparison = result.accuracy_booster?.comparison || {};
  const deltaText = formatScoreDelta(comparison.delta);
  applyTrainingResult(result, {
    statusText: `Boosted model: ${result.best_model}`,
    toastTitle: "Accuracy booster complete",
    toastBody: `${result.best_model} finished with ${result.primary_metric} ${formatNumber(comparison.boosted_score ?? "n/a")} (${deltaText}).`,
  });
  setBusy(false);
}

function buildExamplePredictionRow() {
  const target = document.querySelector("#targetColumn").value;
  const row = {};
  currentProfile.columns.forEach((column) => {
    if (column.name === target) {
      return;
    }
    row[column.name] = column.type === "numeric" ? column.mean || 0 : "";
  });
  return row;
}

function selectedPlaygroundModelId() {
  const select = document.querySelector("#playgroundModelSelect");
  return select?.value || currentModelId || modelRegistry[0]?.model_id || "";
}

function emptyPredictionPlayground(message = "Train a model to unlock the simulator.") {
  const select = document.querySelector("#playgroundModelSelect");
  const summary = document.querySelector("#whatIfSummary");
  const fields = document.querySelector("#whatIfFields");
  const apiBox = document.querySelector("#playgroundApiBox");
  const thresholdPanel = document.querySelector("#playgroundThresholdPanel");
  const payload = document.querySelector("#playgroundPayload");
  const notes = document.querySelector("#apiPlaygroundNotes");

  if (select) {
    select.innerHTML = "";
    select.disabled = true;
  }
  if (summary) {
    summary.className = "summary-row compact empty-state";
    summary.textContent = message;
  }
  if (fields) {
    fields.className = "what-if-fields empty-state";
    fields.textContent = "Feature controls will appear here.";
  }
  if (apiBox) {
    apiBox.textContent = "Select a saved model to build an API request.";
  }
  if (thresholdPanel) {
    thresholdPanel.className = "playground-threshold empty-state";
    thresholdPanel.textContent = "Threshold controls appear for supported classifiers.";
  }
  if (payload) {
    payload.value = "";
  }
  if (notes) {
    notes.className = "deploy-notes empty-state";
    notes.textContent = "Model actions and API notes will appear here.";
  }
  renderDatasetDrift(null);
  renderModelMonitoring(null);
  renderMonitoringOps(null, null);
  renderChampionChallenger(null);
  renderModelApiDocs(null);
  renderDeploymentReadiness(null);
  updateDeployCommandBar();
}

function renderPredictionPlaygroundSelector(models) {
  const select = document.querySelector("#playgroundModelSelect");
  if (!select) {
    return;
  }

  if (!models.length) {
    playgroundMetadata = null;
    emptyPredictionPlayground();
    return;
  }

  const previousSelection = select.value || (models.some((model) => model.model_id === currentModelId) ? currentModelId : "");
  const datasetBackedModel = models.find((model) =>
    datasetRegistry.some((dataset) => dataset.dataset_id === model.dataset_id)
  );
  const selectedModelId = models.some((model) => model.model_id === previousSelection)
    ? previousSelection
    : datasetBackedModel?.model_id
    ? datasetBackedModel.model_id
    : models[0].model_id;

  select.disabled = false;
  select.innerHTML = models
    .map(
      (model) => `
        <option value="${escapeHtml(model.model_id)}">${escapeHtml(model.best_model || "Model")} &middot; ${escapeHtml(model.target_column || "target")} &middot; ${formatNumber(model.rank_score ?? "n/a")}</option>
      `
    )
    .join("");
  select.value = selectedModelId;
  select.onchange = () => loadPredictionPlayground(select.value);

  if (playgroundMetadata?.model_id === selectedModelId) {
    renderPredictionPlayground(playgroundMetadata);
  } else {
    loadPredictionPlayground(selectedModelId);
  }
}

function safeFieldId(name, index) {
  return `whatIfField${index}_${String(name).replace(/[^a-z0-9_-]/gi, "_")}`;
}

function inputValue(value) {
  return value === null || value === undefined ? "" : String(value);
}

function parseWhatIfValue(field, rawValue) {
  if (rawValue === "") {
    return null;
  }
  if (field.type === "numeric") {
    const value = Number(rawValue);
    return Number.isFinite(value) ? value : null;
  }
  return rawValue;
}

function fieldMetaText(field) {
  const parts = [];
  if (field.type && field.type !== "unknown") {
    parts.push(field.type);
  }
  if (typeof field.unique === "number") {
    parts.push(`${formatNumber(field.unique)} unique`);
  }
  if (typeof field.missing_percent === "number" && field.missing_percent > 0) {
    parts.push(`${formatNumber(field.missing_percent)}% missing`);
  }
  if (field.type === "numeric" && (field.min !== undefined || field.max !== undefined)) {
    parts.push(`range ${formatNumber(field.min ?? "n/a")} to ${formatNumber(field.max ?? "n/a")}`);
  }
  return parts.join(" / ") || "required feature";
}

function renderWhatIfInput(field, index) {
  const id = safeFieldId(field.name, index);
  const listId = `${id}Options`;
  const hasOptions = Array.isArray(field.top_values) && field.top_values.length > 0;
  const type = field.type === "numeric" ? "number" : "text";
  const step = field.type === "numeric" ? "step=\"any\"" : "";
  const list = hasOptions ? `list="${escapeHtml(listId)}"` : "";

  return `
    <label class="what-if-field" for="${escapeHtml(id)}">
      <span>${escapeHtml(field.name)}</span>
      <input
        id="${escapeHtml(id)}"
        type="${type}"
        ${step}
        ${list}
        value="${escapeHtml(inputValue(field.example))}"
        data-what-if-input
        data-field-name="${escapeHtml(field.name)}"
      />
      ${hasOptions ? `<datalist id="${escapeHtml(listId)}">${field.top_values.map((value) => `<option value="${escapeHtml(value)}"></option>`).join("")}</datalist>` : ""}
      <small>${escapeHtml(fieldMetaText(field))}</small>
    </label>
  `;
}

function getPlaygroundField(name) {
  return (playgroundMetadata?.fields || []).find((field) => field.name === name) || { name, type: "unknown" };
}

function buildPlaygroundPayloadFromInputs() {
  const row = {};
  document.querySelectorAll("[data-what-if-input]").forEach((input) => {
    const field = getPlaygroundField(input.dataset.fieldName);
    row[field.name] = parseWhatIfValue(field, input.value);
  });

  const payload = { rows: [row] };
  const thresholdInput = document.querySelector("#playgroundThreshold");
  if (thresholdInput) {
    payload.threshold = Number(thresholdInput.value);
  }
  return payload;
}

function syncPlaygroundPayload() {
  const payload = document.querySelector("#playgroundPayload");
  if (!payload || !playgroundMetadata) {
    return;
  }

  const request = buildPlaygroundPayloadFromInputs();
  payload.value = JSON.stringify(request, null, 2);

  const modelStudioPayload = document.querySelector("#predictPayload");
  if (modelStudioPayload && currentModelId === playgroundMetadata.model_id) {
    modelStudioPayload.value = payload.value;
  }
}

function applyWhatIfRow(row) {
  if (!row) {
    return;
  }
  document.querySelectorAll("[data-what-if-input]").forEach((input) => {
    input.value = inputValue(row[input.dataset.fieldName]);
  });
  syncPlaygroundPayload();
}

function renderPlaygroundThreshold(metadata) {
  const panel = document.querySelector("#playgroundThresholdPanel");
  if (!panel) {
    return;
  }

  if (metadata.problem_type !== "classification" || metadata.recommended_threshold === null || metadata.recommended_threshold === undefined) {
    currentThreshold = null;
    panel.className = "playground-threshold empty-state";
    panel.textContent = "This model uses its default prediction threshold.";
    return;
  }

  const threshold = Number(metadata.recommended_threshold || 0.5);
  currentThreshold = threshold;
  panel.className = "playground-threshold";
  panel.innerHTML = `
    <label for="playgroundThreshold">
      <span>Decision threshold</span>
      <strong id="playgroundThresholdValue">${threshold.toFixed(2)}</strong>
    </label>
    <input id="playgroundThreshold" type="range" min="0.2" max="0.8" step="0.01" value="${threshold}" />
    <small>Higher values make positive predictions stricter.</small>
  `;

  document.querySelector("#playgroundThreshold").addEventListener("input", (event) => {
    const value = Number(event.target.value);
    currentThreshold = value;
    document.querySelector("#playgroundThresholdValue").textContent = value.toFixed(2);
    syncPlaygroundPayload();
  });
}

function renderPredictionPlayground(metadata) {
  const summary = document.querySelector("#whatIfSummary");
  const fields = document.querySelector("#whatIfFields");
  const apiBox = document.querySelector("#playgroundApiBox");
  const notes = document.querySelector("#apiPlaygroundNotes");
  if (!summary || !fields || !apiBox || !notes) {
    return;
  }

  currentModelId = metadata.model_id;
  currentPredictionApi = metadata.prediction_api;
  apiBox.textContent = `POST ${metadata.prediction_api}`;
  updateDeployCommandBar();

  summary.className = "summary-row compact";
  summary.innerHTML = `
    <div class="stat"><span>Model</span><strong>${escapeHtml(metadata.best_model)}</strong></div>
    <div class="stat"><span>Target</span><strong>${escapeHtml(metadata.target_column)}</strong></div>
    <div class="stat"><span>Type</span><strong>${escapeHtml(metadata.problem_type)}</strong></div>
    <div class="stat"><span>${escapeHtml(metadata.primary_metric || "Score")}</span><strong>${formatNumber(metadata.rank_score ?? "n/a")}</strong></div>
    <div class="stat"><span>Features</span><strong>${formatNumber(metadata.feature_columns?.length || 0)}</strong></div>
  `;

  fields.className = "what-if-fields";
  fields.innerHTML = (metadata.fields || []).map(renderWhatIfInput).join("");
  fields.querySelectorAll("[data-what-if-input]").forEach((input) => {
    input.addEventListener("input", syncPlaygroundPayload);
  });

  renderPlaygroundThreshold(metadata);
  applyWhatIfRow(metadata.request_template?.rows?.[0] || {});

  const nextActions = metadata.next_actions || [];
  notes.className = nextActions.length ? "deploy-notes" : "deploy-notes empty-state";
  notes.innerHTML = nextActions.length
    ? nextActions.slice(0, 4).map((item) => `<span>${escapeHtml(item)}</span>`).join("")
    : "No model action notes saved yet.";
}

async function loadPredictionPlayground(modelId) {
  if (!modelId) {
    emptyPredictionPlayground();
    return;
  }

  setStatus("Loading prediction playground");
  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/playground`);
  if (!response.ok) {
    emptyPredictionPlayground(await readErrorMessage(response));
    setStatus("Playground unavailable");
    return;
  }

  playgroundMetadata = await response.json();
  playgroundSampleIndex = 0;
  renderPredictionPlayground(playgroundMetadata);
  populateDriftDatasetSelect();
  loadLatestDatasetDrift(modelId);
  loadDeploymentReadiness(modelId);
  loadModelMonitoring(modelId);
  loadChampionChallenger(modelId);
  loadModelApiDocs(modelId);
  setStatus("Prediction playground ready");
}

function useSampleScenario() {
  const rows = playgroundMetadata?.sample_rows || [];
  if (!rows.length) {
    applyWhatIfRow(playgroundMetadata?.request_template?.rows?.[0] || {});
    showToast("Simulator reset", "No dataset sample rows were saved for this model.");
    return;
  }

  const row = rows[playgroundSampleIndex % rows.length];
  playgroundSampleIndex += 1;
  applyWhatIfRow(row);
  showToast("Sample row applied", `Scenario ${playgroundSampleIndex} is ready to score.`);
}

function resetWhatIfInputs() {
  applyWhatIfRow(playgroundMetadata?.request_template?.rows?.[0] || {});
  showToast("Simulator reset", "Default feature values restored.");
}

function readinessClass(status) {
  if (["ready", "review", "blocked"].includes(status)) {
    return status;
  }
  return "unknown";
}

function readinessCheckIcon(status) {
  if (status === "pass") {
    return "OK";
  }
  if (status === "blocked") {
    return "NO";
  }
  if (status === "pending") {
    return "...";
  }
  return "!";
}

function renderDeploymentReadiness(readiness) {
  const container = document.querySelector("#deploymentReadinessChecklist");
  if (!container) {
    return;
  }

  if (!readiness) {
    container.className = "deployment-readiness empty-state";
    container.textContent = "Select a saved model to calculate readiness.";
    return;
  }

  const checks = readiness.checks || [];
  const actions = readiness.actions || [];
  container.className = `deployment-readiness ${readinessClass(readiness.status)}`;
  container.innerHTML = `
    <div class="readiness-head">
      <div>
        <span>${escapeHtml(readiness.status || "review")}</span>
        <strong>${formatNumber(readiness.score ?? 0)}/100</strong>
      </div>
      <b>${escapeHtml(readiness.decision || "Review before deployment")}</b>
    </div>
    <p>${escapeHtml(readiness.summary || "Deployment readiness checks are available.")}</p>
    <div class="readiness-meter"><i style="width:${Math.max(3, Math.min(readiness.score || 0, 100))}%"></i></div>
    <div class="readiness-stats">
      <span><b>${formatNumber(readiness.passed || 0)}</b> passed</span>
      <span><b>${formatNumber(readiness.warnings || 0)}</b> warnings</span>
      <span><b>${formatNumber(readiness.pending || 0)}</b> pending</span>
      <span><b>${formatNumber(readiness.blockers || 0)}</b> blockers</span>
    </div>
    <div class="readiness-grid">
      ${checks
        .map(
          (check) => `
            <article class="readiness-check ${escapeHtml(check.status)}">
              <div>
                <span>${escapeHtml(readinessCheckIcon(check.status))}</span>
                <strong>${escapeHtml(check.label)}</strong>
              </div>
              <p>${escapeHtml(check.detail)}</p>
              <small>${escapeHtml(check.action)}</small>
            </article>
          `
        )
        .join("")}
    </div>
    ${
      actions.length
        ? `<div class="readiness-actions">
            ${actions.map((action) => `<span>${escapeHtml(action)}</span>`).join("")}
          </div>`
        : ""
    }
  `;
}

async function loadDeploymentReadiness(modelId = selectedPlaygroundModelId()) {
  if (!modelId) {
    renderDeploymentReadiness(null);
    return;
  }

  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/readiness`);
  if (!response.ok) {
    renderDeploymentReadiness({
      status: "blocked",
      score: 0,
      decision: "Readiness unavailable",
      summary: await readErrorMessage(response),
      checks: [],
      actions: [],
      passed: 0,
      warnings: 0,
      pending: 0,
      blockers: 1,
    });
    return;
  }

  const body = await response.json();
  renderDeploymentReadiness(body.readiness);
}

function monitoringClass(status) {
  if (["healthy", "watch", "review", "critical"].includes(status)) {
    return status;
  }
  return "unknown";
}

function monitoringLabel(status) {
  const labels = {
    healthy: "Healthy",
    watch: "Watch",
    review: "Needs review",
    critical: "Critical",
  };
  return labels[status] || "Unknown";
}

function monitoringCardIcon(status) {
  if (status === "healthy") {
    return "OK";
  }
  if (status === "critical") {
    return "NO";
  }
  return "!";
}

function renderSignalMeter(label, value) {
  const score = Math.max(0, Math.min(Number(value) || 0, 100));
  return `
    <div class="monitoring-signal">
      <span>${escapeHtml(label)}</span>
      <div class="monitoring-track"><i style="width:${Math.max(4, score)}%"></i></div>
      <strong>${formatNumber(score)}</strong>
    </div>
  `;
}

function renderModelMonitoring(monitoring) {
  const container = document.querySelector("#modelMonitoringDashboard");
  if (!container) {
    return;
  }

  currentMonitoringDashboard = monitoring || null;
  if (!monitoring) {
    container.className = "monitoring-dashboard empty-state";
    container.textContent = "Select a saved model to load monitoring signals.";
    updateDeployCommandBar();
    return;
  }

  const recommendation = monitoring.recommendation || {};
  const signals = monitoring.signals || {};
  const cards = monitoring.cards || [];
  const timeline = monitoring.timeline || [];
  const actions = monitoring.actions || [];
  container.className = `monitoring-dashboard ${monitoringClass(monitoring.status)}`;
  container.innerHTML = `
    <div class="monitoring-head">
      <div class="monitoring-score">
        <span>${escapeHtml(monitoringLabel(monitoring.status))}</span>
        <strong>${formatNumber(monitoring.score ?? 0)}/100</strong>
      </div>
      <div class="monitoring-recommendation ${escapeHtml(recommendation.level || "watch")}">
        <span>${escapeHtml(recommendation.retrain_recommended ? "Retrain recommended" : "Next action")}</span>
        <b>${escapeHtml(recommendation.title || "Review monitoring signals")}</b>
        <p>${escapeHtml(recommendation.detail || monitoring.summary || "Monitoring signals are ready.")}</p>
      </div>
    </div>

    <div class="monitoring-card-grid">
      ${cards
        .map(
          (card) => `
            <article class="monitoring-card ${escapeHtml(card.status || "review")}">
              <div>
                <span>${escapeHtml(monitoringCardIcon(card.status))}</span>
                <strong>${escapeHtml(card.label)}</strong>
              </div>
              <b>${escapeHtml(card.value)}</b>
              <p>${escapeHtml(card.detail)}</p>
            </article>
          `
        )
        .join("")}
    </div>

    <div class="monitoring-lower-grid">
      <article class="monitoring-signals">
        <strong>Signal breakdown</strong>
        ${renderSignalMeter("Readiness", signals.readiness_score)}
        ${renderSignalMeter("Trust", signals.trust_score)}
        ${renderSignalMeter("Drift", signals.drift_score)}
        ${renderSignalMeter("Confidence", signals.confidence_score)}
        ${renderSignalMeter("Audit", signals.audit_score)}
      </article>

      <article class="monitoring-timeline">
        <strong>Recent monitor events</strong>
        ${
          timeline.length
            ? timeline
                .map(
                  (event) => `
                    <div class="monitoring-event ${escapeHtml(event.status || "info")}">
                      <span>${escapeHtml(event.kind || "event")}</span>
                      <div>
                        <b>${escapeHtml(event.title || "Event")}</b>
                        <small>${escapeHtml(event.detail || "")}</small>
                      </div>
                      <time>${escapeHtml(formatAuditTime(event.created_at))}</time>
                    </div>
                  `
                )
                .join("")
            : "<p>No monitoring events yet.</p>"
        }
      </article>
    </div>

    ${
      actions.length
        ? `<div class="monitoring-actions">
            ${actions.map((action) => `<span>${escapeHtml(action)}</span>`).join("")}
          </div>`
        : ""
    }
  `;
  updateDeployCommandBar();
  renderSidebarPulse();
  renderStudioOverview();
}

async function loadModelMonitoring(modelId = selectedPlaygroundModelId()) {
  if (!modelId) {
    renderModelMonitoring(null);
    renderMonitoringOps(null, null);
    return;
  }

  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/monitoring`);
  if (!response.ok) {
    renderModelMonitoring({
      status: "critical",
      score: 0,
      recommendation: {
        level: "critical",
        title: "Monitoring unavailable",
        detail: await readErrorMessage(response),
        action: "Refresh after retraining or restoring this model.",
      },
      cards: [],
      timeline: [],
      signals: {},
      actions: [],
    });
    renderMonitoringOps(null, null);
    return;
  }

  const body = await response.json();
  renderModelMonitoring(body.monitoring);
  renderMonitoringOps(body.history, body.alerts);
}

function trendValue(snapshot, key) {
  const value = snapshot?.[key];
  return typeof value === "number" ? value : null;
}

function renderTrendChart(title, key, snapshots) {
  const values = snapshots.map((snapshot) => trendValue(snapshot, key)).filter((value) => value !== null);
  const latest = values.length ? values[values.length - 1] : null;
  const first = values.length ? values[0] : null;
  const delta = latest !== null && first !== null ? latest - first : null;
  const width = 260;
  const height = 76;
  const pad = 10;
  const usableWidth = width - pad * 2;
  const usableHeight = height - pad * 2;
  const points = values.length
    ? values
        .map((value, index) => {
          const x = values.length === 1 ? width / 2 : pad + (index / (values.length - 1)) * usableWidth;
          const y = pad + (1 - Math.max(0, Math.min(value, 100)) / 100) * usableHeight;
          return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ")
    : "";

  return `
    <article class="trend-card">
      <div>
        <span>${escapeHtml(title)}</span>
        <strong>${latest === null ? "n/a" : `${formatNumber(latest)}/100`}</strong>
      </div>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)} trend">
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />
        ${points ? `<polyline points="${points}" />` : ""}
      </svg>
      <small>${delta === null ? "Waiting for history" : `${formatScoreDelta(delta)} since first snapshot`}</small>
    </article>
  `;
}

function renderMonitoringHistory(history) {
  const container = document.querySelector("#monitoringHistoryCharts");
  if (!container) {
    return;
  }

  const snapshots = history?.snapshots || [];
  const summary = history?.summary || {};
  if (!snapshots.length) {
    container.className = "monitoring-history empty-state";
    container.textContent = "Monitoring trend charts appear after the first health snapshot.";
    return;
  }

  container.className = "monitoring-history";
  container.innerHTML = `
    <div class="history-head">
      <div>
        <span>Snapshots</span>
        <strong>${formatNumber(summary.filtered_snapshots ?? snapshots.length)}</strong>
      </div>
      <p>${summary.score_delta === null || summary.score_delta === undefined ? "Trend is building from live monitoring snapshots." : `Latest health moved ${formatScoreDelta(summary.score_delta)} from the previous snapshot.`}</p>
    </div>
    <div class="trend-grid">
      ${renderTrendChart("Health", "score", snapshots)}
      ${renderTrendChart("Readiness", "readiness_score", snapshots)}
      ${renderTrendChart("Trust", "trust_score", snapshots)}
      ${renderTrendChart("Drift", "drift_score", snapshots)}
      ${renderTrendChart("Confidence", "confidence_score", snapshots)}
      ${renderTrendChart("Audit", "audit_score", snapshots)}
    </div>
  `;
}

function alertLevelLabel(level) {
  const labels = {
    critical: "Critical",
    warning: "Warning",
    info: "Info",
  };
  return labels[level] || "Info";
}

function renderAlertCenter(alertState) {
  const container = document.querySelector("#alertCenter");
  if (!container) {
    return;
  }

  const alerts = alertState?.alerts || [];
  const summary = alertState?.summary || {};
  if (!alerts.length) {
    container.className = "alert-center empty-state";
    container.textContent = "No active alerts for the selected model.";
    return;
  }

  container.className = "alert-center";
  container.innerHTML = `
    <div class="alert-summary">
      <div class="stat"><span>Critical</span><strong>${formatNumber(summary.critical ?? 0)}</strong></div>
      <div class="stat"><span>Warnings</span><strong>${formatNumber(summary.warning ?? 0)}</strong></div>
      <div class="stat"><span>Info</span><strong>${formatNumber(summary.info ?? 0)}</strong></div>
    </div>
    <div class="alert-list">
      ${alerts
        .map(
          (alert) => `
            <article class="alert-card ${escapeHtml(alert.level || "info")}">
              <div>
                <span>${escapeHtml(alertLevelLabel(alert.level))}</span>
                <small>${escapeHtml(alert.source || "monitoring")} &middot; ${escapeHtml(formatAuditTime(alert.last_seen || alert.created_at))}</small>
              </div>
              <strong>${escapeHtml(alert.title || "Monitoring alert")}</strong>
              <p>${escapeHtml(alert.detail || "")}</p>
              <b>${escapeHtml(alert.action || "Review this signal.")}</b>
            </article>
          `
        )
        .join("")}
    </div>
  `;
}

function renderMonitoringOps(history, alertState) {
  currentMonitoringHistory = history || null;
  currentAlertCenter = alertState || null;
  renderMonitoringHistory(history);
  renderAlertCenter(alertState);
  updateDeployCommandBar();
}

function championStatusText(status) {
  const labels = {
    production: "Production",
    champion: "Champion",
    challenger: "Challenger",
    unscored: "Unscored",
  };
  return labels[status] || "Challenger";
}

function championDelta(value) {
  if (typeof value !== "number") {
    return "n/a";
  }
  return formatScoreDelta(value);
}

function renderChampionMini(model, label) {
  if (!model) {
    return `
      <article class="champion-mini empty-state">
        <span>${escapeHtml(label)}</span>
        <strong>No model</strong>
      </article>
    `;
  }

  return `
    <article class="champion-mini">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(model.best_model || "Model")}</strong>
      <p>${escapeHtml(model.target_column || "target")} &middot; ${escapeHtml(model.primary_metric || "score")}</p>
      <div>
        <b>${formatNumber(model.rank_score ?? "n/a")}</b>
        <small>${model.rank ? `rank #${formatNumber(model.rank)}` : "rank n/a"}</small>
      </div>
    </article>
  `;
}

function renderChampionChallenger(comparison) {
  const container = document.querySelector("#championChallengerBoard");
  const retrainButton = document.querySelector("#retrainFromDrift");
  const promoteButton = document.querySelector("#promoteToProduction");
  if (!container) {
    return;
  }

  currentChampionComparison = comparison || null;
  if (retrainButton) {
    retrainButton.disabled = !selectedPlaygroundModelId();
  }
  if (promoteButton) {
    promoteButton.disabled = !selectedPlaygroundModelId();
  }

  if (!comparison) {
    container.className = "champion-board empty-state";
    container.textContent = "Select a saved model to compare champion and challenger runs.";
    updateDeployCommandBar();
    return;
  }

  const champion = comparison.champion;
  const production = comparison.production_model;
  const promotion = comparison.promotion || {};
  const selected = comparison.selected_model || {};
  const challengers = comparison.challengers || [];
  const actions = comparison.actions || [];
  container.className = `champion-board ${escapeHtml(comparison.status || "challenger")}`;
  container.innerHTML = `
    <div class="champion-head">
      <div>
        <span>${escapeHtml(championStatusText(comparison.status))}</span>
        <strong>${selected.rank ? `#${formatNumber(selected.rank)}` : "n/a"}</strong>
      </div>
      <p>
        ${escapeHtml(selected.best_model || "Selected model")}
        is ${escapeHtml(championDelta(selected.score_delta_vs_champion))}
        from the champion on ${escapeHtml(selected.primary_metric || "score")}.
      </p>
      <span class="metric-pill">${formatNumber(comparison.group_size || 0)} comparable run(s)</span>
    </div>

    <div class="champion-compare-grid">
      ${renderChampionMini(production, "Production")}
      ${renderChampionMini(champion, "Champion")}
      ${renderChampionMini(selected, "Selected model")}
    </div>
    ${
      production
        ? `<div class="production-note">
            <span>Production endpoint</span>
            <code>${escapeHtml(production.prediction_api || "")}</code>
            <small>Promoted ${escapeHtml(formatAuditTime(promotion.promoted_at))}</small>
          </div>`
        : ""
    }

    <div class="challenger-list">
      <strong>Top challengers</strong>
      ${
        challengers.length
          ? challengers
              .map(
                (item) => `
                  <article class="challenger-row ${item.model_id === selected.model_id ? "selected" : ""}">
                    <div>
                      <span>#${formatNumber(item.rank || 0)}</span>
                      <strong>${escapeHtml(item.best_model || "Model")}</strong>
                      <small>${escapeHtml(item.created_at || "")}</small>
                    </div>
                    <b>${formatNumber(item.rank_score ?? "n/a")}</b>
                    <em>${escapeHtml(championDelta(item.score_delta_vs_champion))}</em>
                  </article>
                `
              )
              .join("")
          : "<p>No challengers yet. Retrain from drift or train another model with the same target.</p>"
      }
    </div>

    ${
      actions.length
        ? `<div class="champion-guidance">
            ${actions.map((action) => `<span>${escapeHtml(action)}</span>`).join("")}
          </div>`
        : ""
    }
  `;
  updateDeployCommandBar();
}

async function loadChampionChallenger(modelId = selectedPlaygroundModelId()) {
  if (!modelId) {
    renderChampionChallenger(null);
    return;
  }

  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/challengers`);
  if (!response.ok) {
    renderChampionChallenger({
      status: "unscored",
      selected_model: {},
      champion: null,
      challengers: [],
      group_size: 0,
      actions: [await readErrorMessage(response)],
    });
    return;
  }

  const body = await response.json();
  renderChampionChallenger(body.champion_challenger);
}

async function runPromoteToProduction() {
  const modelId = selectedPlaygroundModelId();
  if (!modelId) {
    alert("Select a saved model first.");
    return;
  }

  setStatus("Promoting model to production");
  setBusy(true);
  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/promote`, {
    method: "POST",
  });
  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Promotion failed");
    setBusy(false);
    return;
  }

  const body = await response.json();
  renderChampionChallenger(body.champion_challenger);
  await loadModelRegistry();
  await loadModelMonitoring(modelId);
  await loadModelApiDocs(modelId);
  setStatus("Model promoted to production");
  showToast("Production model updated", `${body.promotion.best_model || "Model"} is now production for ${body.promotion.target_column}.`);
  setBusy(false);
}

async function runRetrainFromDrift() {
  const modelId = selectedPlaygroundModelId();
  if (!modelId) {
    alert("Select a saved model first.");
    return;
  }

  const datasetId = currentDriftReport?.comparison_dataset_id || document.querySelector("#driftDatasetSelect")?.value || null;
  setStatus("Retraining from drift dataset");
  setBusy(true);
  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/retrain-from-drift`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId }),
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Retrain from drift failed");
    setBusy(false);
    return;
  }

  const body = await response.json();
  const result = body.training_result;
  applyTrainingResult(result, {
    statusText: `Drift retrain complete: ${result.best_model}`,
    toastTitle: "Retrain complete",
    toastBody: `${result.best_model} is saved as a new challenger for ${result.target_column}.`,
  });
  renderChampionChallenger(body.champion_challenger);
  await loadModelRegistry();
  await loadPredictionPlayground(result.model_id);
  setBusy(false);
}

function renderDocsFieldList(fields = []) {
  if (!fields.length) {
    return "<span class=\"muted\">No required feature list saved.</span>";
  }
  return fields.map((field) => `<span>${escapeHtml(field)}</span>`).join("");
}

function renderResponseFields(fields = []) {
  return fields
    .map(
      (field) => `
        <tr>
          <td>${escapeHtml(field.name)}</td>
          <td>${escapeHtml(field.type)}</td>
          <td>${escapeHtml(field.description)}</td>
        </tr>
      `
    )
    .join("");
}

function attachApiDocCopyButtons(container) {
  container.querySelectorAll("[data-copy-doc]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = container.querySelector(`#${button.dataset.copyDoc}`);
      if (!target) {
        return;
      }
      try {
        await navigator.clipboard.writeText(target.textContent || "");
        showToast("Copied", `${button.textContent.trim()} copied.`);
      } catch {
        showToast("Copy unavailable", "Select the snippet text and copy it manually.");
      }
    });
  });
}

function renderModelApiDocs(docs) {
  const container = document.querySelector("#modelApiDocs");
  if (!container) {
    return;
  }

  currentApiDocs = docs || null;
  if (!docs) {
    container.className = "api-docs empty-state";
    container.textContent = "Select a saved model to generate API documentation.";
    return;
  }

  container.className = "api-docs";
  container.innerHTML = `
    <div class="api-docs-head">
      <div>
        <span>${escapeHtml(docs.is_production ? "Production API" : "Model API")}</span>
        <strong>${escapeHtml(docs.method)} ${escapeHtml(docs.endpoint)}</strong>
        <p>${escapeHtml(docs.model_name || "Model")} predicts ${escapeHtml(docs.target_column || "target")} using ${escapeHtml(docs.problem_type || "problem")} scoring.</p>
      </div>
      <div class="api-docs-score">
        <span>${escapeHtml(docs.primary_metric || "score")}</span>
        <strong>${formatNumber(docs.rank_score ?? "n/a")}</strong>
      </div>
    </div>

    <div class="api-docs-grid">
      <article>
        <strong>Required features</strong>
        <div class="api-feature-list">${renderDocsFieldList(docs.required_features || [])}</div>
      </article>
      <article>
        <strong>Request schema</strong>
        <p><b>rows</b>: ${escapeHtml(docs.request_schema?.rows || "")}</p>
        <p><b>threshold</b>: ${escapeHtml(docs.request_schema?.threshold || "")}</p>
      </article>
    </div>

    <div class="api-snippet-grid">
      <article class="api-snippet">
        <div>
          <strong>Sample payload</strong>
        </div>
        <pre>${escapeHtml(JSON.stringify(docs.sample_payload || {}, null, 2))}</pre>
      </article>
      <article class="api-snippet">
        <div>
          <strong>Response example</strong>
        </div>
        <pre>${escapeHtml(JSON.stringify(docs.response_example || {}, null, 2))}</pre>
      </article>
      <article class="api-snippet">
        <div>
          <strong>curl</strong>
          <button class="ghost-button" type="button" data-copy-doc="curlSnippet">Copy curl</button>
        </div>
        <pre id="curlSnippet">${escapeHtml(docs.curl || "")}</pre>
      </article>
      <article class="api-snippet">
        <div>
          <strong>Python</strong>
          <button class="ghost-button" type="button" data-copy-doc="pythonSnippet">Copy Python</button>
        </div>
        <pre id="pythonSnippet">${escapeHtml(docs.python || "")}</pre>
      </article>
    </div>

    <div class="table-wrap compact-table">
      <table>
        <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
        <tbody>${renderResponseFields(docs.response_fields || [])}</tbody>
      </table>
    </div>

    <div class="api-doc-notes">
      ${(docs.notes || []).map((note) => `<span>${escapeHtml(note)}</span>`).join("")}
    </div>
  `;
  attachApiDocCopyButtons(container);
}

async function loadModelApiDocs(modelId = selectedPlaygroundModelId()) {
  if (!modelId) {
    renderModelApiDocs(null);
    return;
  }

  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/api-docs`);
  if (!response.ok) {
    renderModelApiDocs(null);
    return;
  }
  const body = await response.json();
  renderModelApiDocs(body.api_docs);
}

function driftClass(status) {
  if (["low", "medium", "high", "blocked"].includes(status)) {
    return status;
  }
  return "unknown";
}

function driftStatusLabel(status) {
  const labels = {
    low: "Low drift",
    medium: "Medium drift",
    high: "High drift",
    blocked: "Blocked",
  };
  return labels[status] || "Not checked";
}

function driftCheckIcon(status) {
  if (status === "pass") {
    return "OK";
  }
  if (status === "blocked") {
    return "NO";
  }
  return "!";
}

function driftCell(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  return typeof value === "number" ? `${formatNumber(value)}${suffix}` : escapeHtml(value);
}

function driftSeverity(value) {
  return `<span class="drift-severity ${escapeHtml(value || "low")}">${escapeHtml(value || "low")}</span>`;
}

function renderDriftTable(title, rows, columns) {
  if (!rows?.length) {
    return `
      <article class="drift-table-card">
        <strong>${escapeHtml(title)}</strong>
        <p>No notable shifts found.</p>
      </article>
    `;
  }

  return `
    <article class="drift-table-card">
      <strong>${escapeHtml(title)}</strong>
      <div class="table-wrap compact-table">
        <table>
          <thead>
            <tr>${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}</tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (row) => `
                  <tr>
                    ${columns
                      .map((column) => {
                        const value = row[column.key];
                        if (column.key === "severity") {
                          return `<td>${driftSeverity(value)}</td>`;
                        }
                        return `<td>${driftCell(value, column.suffix || "")}</td>`;
                      })
                      .join("")}
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </article>
  `;
}

function renderDatasetDrift(drift) {
  const container = document.querySelector("#datasetDriftMonitor");
  if (!container) {
    return;
  }

  currentDriftReport = drift || null;
  if (!drift) {
    container.className = "dataset-drift empty-state";
    container.textContent = "Run a drift check to compare the selected model training data with another dataset.";
    return;
  }

  const checks = drift.checks || [];
  const actions = drift.actions || [];
  const rowCounts = drift.row_counts || {};
  const columnCounts = drift.column_counts || {};
  container.className = `dataset-drift ${driftClass(drift.status)}`;
  container.innerHTML = `
    <div class="drift-head">
      <div>
        <span>${escapeHtml(driftStatusLabel(drift.status))}</span>
        <strong>${formatNumber(drift.score ?? 0)}/100</strong>
      </div>
      <p>${escapeHtml(drift.summary || "Dataset drift report is ready.")}</p>
    </div>

    <div class="drift-metrics">
      <div class="stat"><span>Reference rows</span><strong>${formatNumber(rowCounts.reference ?? 0)}</strong></div>
      <div class="stat"><span>Compare rows</span><strong>${formatNumber(rowCounts.comparison ?? 0)}</strong></div>
      <div class="stat"><span>Features checked</span><strong>${formatNumber(columnCounts.compared_features ?? 0)}</strong></div>
      <div class="stat"><span>Required features</span><strong>${formatNumber(columnCounts.required_features ?? 0)}</strong></div>
    </div>

    <div class="drift-checks">
      ${checks
        .map(
          (check) => `
            <article class="drift-check ${escapeHtml(check.status)}">
              <div>
                <span>${escapeHtml(driftCheckIcon(check.status))}</span>
                <strong>${escapeHtml(check.label)}</strong>
              </div>
              <p>${escapeHtml(check.detail)}</p>
              <small>${escapeHtml(check.action)}</small>
            </article>
          `
        )
        .join("")}
    </div>

    <div class="drift-tables">
      ${renderDriftTable("Missing-value drift", drift.missing_drift || [], [
        { key: "column", label: "Column" },
        { key: "reference_missing_percent", label: "Training", suffix: "%" },
        { key: "comparison_missing_percent", label: "Compare", suffix: "%" },
        { key: "shift_points", label: "Shift", suffix: " pts" },
        { key: "severity", label: "Risk" },
      ])}
      ${renderDriftTable("Numeric drift", drift.numeric_drift || [], [
        { key: "column", label: "Column" },
        { key: "reference_mean", label: "Training mean" },
        { key: "comparison_mean", label: "Compare mean" },
        { key: "mean_shift", label: "Shift" },
        { key: "severity", label: "Risk" },
      ])}
      ${renderDriftTable("Category drift", drift.categorical_drift || [], [
        { key: "column", label: "Column" },
        { key: "distribution_shift", label: "Dist shift" },
        { key: "unseen_category_share", label: "Unseen", suffix: "%" },
        { key: "reference_top", label: "Training top" },
        { key: "comparison_top", label: "Compare top" },
        { key: "severity", label: "Risk" },
      ])}
    </div>

    ${
      actions.length
        ? `<div class="drift-actions">
            ${actions.map((action) => `<span>${escapeHtml(action)}</span>`).join("")}
          </div>`
        : ""
    }
  `;
}

async function loadLatestDatasetDrift(modelId = selectedPlaygroundModelId()) {
  if (!modelId) {
    renderDatasetDrift(null);
    return;
  }

  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/drift/latest`);
  if (!response.ok) {
    renderDatasetDrift(null);
    return;
  }
  const body = await response.json();
  renderDatasetDrift(body.drift);
}

async function runDatasetDriftCheck() {
  const modelId = selectedPlaygroundModelId();
  const datasetId = document.querySelector("#driftDatasetSelect")?.value;
  if (!modelId) {
    alert("Select a saved model first.");
    return;
  }
  if (!datasetId) {
    alert("Select a comparison dataset first.");
    return;
  }

  setStatus("Running dataset drift check");
  setBusy(true);
  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/drift`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_id: datasetId }),
  });
  if (!response.ok) {
    alert(await readErrorMessage(response));
    setBusy(false);
    setStatus("Drift check failed");
    return;
  }

  const body = await response.json();
  renderDatasetDrift(body.drift);
  await loadDeploymentReadiness(modelId);
  await loadModelMonitoring(modelId);
  await loadChampionChallenger(modelId);
  setBusy(false);
  setStatus("Dataset drift report ready");
  showToast("Drift check complete", `${driftStatusLabel(body.drift?.status)} with score ${formatNumber(body.drift?.score ?? 0)}/100.`);
}

function renderConfidenceDistribution(distribution = [], totalRows = 0) {
  if (!distribution.length) {
    return "";
  }

  return `
    <div class="prediction-distribution">
      ${distribution
        .map((item) => {
          const share = totalRows ? Math.max(6, (item.count / totalRows) * 100) : 8;
          return `
            <div>
              <span>${escapeHtml(item.label)}</span>
              <strong>${formatNumber(item.count)}</strong>
              <div class="importance-track"><div style="width:${Math.min(share, 100)}%"></div></div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderProbabilityBars(probabilities = {}) {
  const entries = Object.entries(probabilities);
  if (!entries.length) {
    return "<span class=\"muted\">No probability output available.</span>";
  }

  return entries
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .map(
      ([label, value]) => `
        <div class="probability-row">
          <span>${escapeHtml(label)}</span>
          <div class="importance-track"><div style="width:${Math.max(5, Math.min(Number(value) * 100, 100))}%"></div></div>
          <strong>${formatNumber(Number(value) * 100)}%</strong>
        </div>
      `
    )
    .join("");
}

function renderPredictionConfidenceDashboard(body) {
  const container = document.querySelector("#predictionConfidenceDashboard");
  if (!container) {
    return;
  }

  const summary = body?.confidence_summary || {};
  const details = body?.prediction_details || [];
  const lowRows = summary.low_confidence_row_numbers || [];

  container.className = "prediction-confidence";
  container.innerHTML = `
    <div class="prediction-confidence-head">
      <div>
        <span class="panel-kicker">Prediction Confidence</span>
        <h2>${summary.available ? "Confidence dashboard" : "Prediction results"}</h2>
        <p class="muted">${escapeHtml(summary.recommendation || summary.message || "Prediction results are ready.")}</p>
      </div>
      <span class="metric-pill">${formatNumber(summary.total_rows ?? details.length)} rows</span>
    </div>

    <div class="prediction-confidence-grid">
      <div class="stat"><span>Average confidence</span><strong>${summary.available ? `${formatNumber((summary.average_confidence || 0) * 100)}%` : "n/a"}</strong></div>
      <div class="stat"><span>Minimum confidence</span><strong>${summary.available ? `${formatNumber((summary.minimum_confidence || 0) * 100)}%` : "n/a"}</strong></div>
      <div class="stat"><span>High confidence</span><strong>${formatNumber(summary.high_confidence_rows ?? 0)}</strong></div>
      <div class="stat"><span>Medium confidence</span><strong>${formatNumber(summary.medium_confidence_rows ?? 0)}</strong></div>
      <div class="stat ${summary.low_confidence_rows ? "warning" : ""}"><span>Low confidence</span><strong>${formatNumber(summary.low_confidence_rows ?? 0)}</strong></div>
    </div>

    ${
      lowRows.length
        ? `<div class="confidence-alert">Review low-confidence row${lowRows.length > 1 ? "s" : ""}: ${lowRows.map(formatNumber).join(", ")}</div>`
        : ""
    }

    ${renderConfidenceDistribution(summary.prediction_distribution || [], summary.total_rows || details.length)}

    <div class="prediction-row-list">
      ${details
        .map(
          (item) => `
            <article class="prediction-row-card ${item.confidence_band || "unavailable"}">
              <div class="prediction-row-top">
                <span>Row ${formatNumber(item.row)}</span>
                <b>${escapeHtml(item.confidence_band || "unavailable")}</b>
              </div>
              <div class="prediction-value">
                <span>Prediction</span>
                <strong>${escapeHtml(item.prediction)}</strong>
              </div>
              ${
                typeof item.confidence === "number"
                  ? `<div class="prediction-value">
                      <span>Confidence</span>
                      <strong>${formatNumber(item.confidence * 100)}%</strong>
                    </div>`
                  : ""
              }
              <div class="probability-list">${renderProbabilityBars(item.probabilities || {})}</div>
              ${
                item.low_confidence
                  ? `<p class="muted">Low confidence. Check this row before production use.</p>`
                  : ""
              }
            </article>
          `
        )
        .join("")}
    </div>
  `;
}

function renderBatchPredictionStatus(filename) {
  const container = document.querySelector("#predictionConfidenceDashboard");
  if (!container) {
    return;
  }
  container.className = "prediction-confidence";
  container.innerHTML = `
    <div class="prediction-confidence-head">
      <div>
        <span class="panel-kicker">Batch Prediction</span>
        <h2>CSV predictions downloaded</h2>
        <p class="muted">The downloaded file includes prediction, confidence, confidence band, low-confidence flag, and class probability columns when the model supports them.</p>
      </div>
      <span class="metric-pill">${escapeHtml(filename)}</span>
    </div>
  `;
}

function formatAuditTime(value) {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return escapeHtml(value);
  }
  return new Intl.DateTimeFormat("en-IN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function auditPredictionSummary(audit) {
  const predictions = audit.predictions || [];
  if (!predictions.length) {
    return "No prediction saved";
  }
  const unique = [...new Set(predictions.map((value) => String(value)))];
  if (unique.length === 1) {
    return `${escapeHtml(unique[0])}${predictions.length > 1 ? ` (${formatNumber(predictions.length)} rows)` : ""}`;
  }
  return `${formatNumber(unique.length)} outcomes across ${formatNumber(predictions.length)} rows`;
}

function auditConfidenceText(audit) {
  const summary = audit.confidence_summary || {};
  if (typeof summary.average_confidence === "number") {
    return `${formatNumber(summary.average_confidence * 100)}% avg confidence`;
  }
  return summary.recommendation || "Confidence unavailable";
}

function renderPredictionAuditLog(body) {
  const container = document.querySelector("#predictionAuditLog");
  if (!container) {
    return;
  }

  const audits = body?.audits || [];
  const summary = body?.summary || {};
  if (!audits.length) {
    container.className = "prediction-audit-log empty-state";
    container.textContent = "Run a prediction to build the audit log.";
    return;
  }

  container.className = "prediction-audit-log";
  container.innerHTML = `
    <div class="audit-summary-strip">
      <div class="stat"><span>Visible logs</span><strong>${formatNumber(summary.filtered_audits ?? audits.length)}</strong></div>
      <div class="stat"><span>Total logs</span><strong>${formatNumber(summary.total_audits ?? audits.length)}</strong></div>
      <div class="stat"><span>Low confidence rows</span><strong>${formatNumber(summary.low_confidence_rows ?? 0)}</strong></div>
      <div class="stat"><span>Avg confidence</span><strong>${typeof summary.average_confidence === "number" ? `${formatNumber(summary.average_confidence * 100)}%` : "n/a"}</strong></div>
    </div>
    <div class="audit-card-grid">
      ${audits
        .map((audit) => {
          const confidenceSummary = audit.confidence_summary || {};
          const lowCount = confidenceSummary.low_confidence_rows || 0;
          const details = audit.prediction_details || [];
          return `
            <article class="audit-card ${lowCount ? "warning" : ""}">
              <div class="audit-card-top">
                <span>${escapeHtml(audit.request_type || "single")}</span>
                <b>${formatAuditTime(audit.created_at)}</b>
              </div>
              <strong>${escapeHtml(audit.best_model || "Saved model")}</strong>
              <p>${escapeHtml(audit.target_column || "target")} &middot; ${escapeHtml(audit.problem_type || "problem")} &middot; ${formatNumber(audit.row_count || 0)} row(s)</p>
              <div class="audit-result">
                <span>Prediction</span>
                <strong>${auditPredictionSummary(audit)}</strong>
              </div>
              <div class="audit-meta">
                <span>${escapeHtml(auditConfidenceText(audit))}</span>
                ${lowCount ? `<span>${formatNumber(lowCount)} low confidence</span>` : "<span>No low-confidence rows</span>"}
              </div>
              <details>
                <summary>Review input and output</summary>
                <div class="audit-detail-grid">
                  <div>
                    <span>Input preview</span>
                    <pre>${escapeHtml(JSON.stringify(audit.input_preview || [], null, 2))}</pre>
                  </div>
                  <div>
                    <span>Prediction details</span>
                    <pre>${escapeHtml(JSON.stringify(details, null, 2))}</pre>
                  </div>
                </div>
              </details>
            </article>
          `;
        })
        .join("")}
    </div>
  `;
}

async function loadPredictionAuditLog() {
  const response = await fetch("/api/prediction-audits?limit=20");
  if (!response.ok) {
    return;
  }

  const body = await response.json();
  renderPredictionAuditLog(body);
}

async function runPrediction() {
  if (!currentPredictionApi) {
    alert("Train a model first.");
    return;
  }

  let payload;
  try {
    payload = JSON.parse(document.querySelector("#predictPayload").value);
  } catch {
    alert("Prediction payload must be valid JSON.");
    return;
  }
  if (currentThreshold !== null) {
    payload.threshold = currentThreshold;
  }

  setStatus("Running prediction");
  setBusy(true);
  const response = await fetch(currentPredictionApi, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Prediction failed");
    setBusy(false);
    return;
  }

  const body = await response.json();
  renderPredictionConfidenceDashboard(body);
  document.querySelector("#predictionOutput").textContent = JSON.stringify(body, null, 2);
  loadPredictionAuditLog();
  loadDeploymentReadiness(currentModelId);
  loadModelMonitoring(currentModelId);
  setStatus("Prediction ready");
  setBusy(false);
  showView("api");
  showDeployTab("results");
}

async function runPlaygroundPrediction() {
  const modelId = selectedPlaygroundModelId();
  const payloadText = document.querySelector("#playgroundPayload")?.value || "";
  if (!modelId || !playgroundMetadata) {
    alert("Select a saved model first.");
    return;
  }

  let payload;
  try {
    payload = JSON.parse(payloadText);
  } catch {
    alert("API playground payload must be valid JSON.");
    return;
  }

  if (!Array.isArray(payload.rows) || !payload.rows.length) {
    alert("Payload must include at least one row.");
    return;
  }

  setStatus("Running what-if prediction");
  setBusy(true);
  const response = await fetch(`/api/predict/${encodeURIComponent(modelId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("What-if prediction failed");
    setBusy(false);
    return;
  }

  const body = await response.json();
  renderPredictionConfidenceDashboard(body);
  document.querySelector("#predictionOutput").textContent = JSON.stringify(body, null, 2);
  document.querySelector("#predictPayload").value = JSON.stringify(payload, null, 2);
  currentModelId = modelId;
  currentPredictionApi = playgroundMetadata.prediction_api;
  loadPredictionAuditLog();
  loadDeploymentReadiness(modelId);
  loadModelMonitoring(modelId);
  setStatus("What-if prediction ready");
  setBusy(false);
  showDeployTab("results");
  showToast("Prediction ready", `${playgroundMetadata.best_model} scored the current what-if row.`);
}

function getDownloadFilename(response, fallback) {
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/i);
  return match?.[1] || fallback;
}

async function runBatchPrediction() {
  if (!currentModelId) {
    alert("Train a model first.");
    return;
  }

  const file = batchPredictFileInput?.files?.[0];
  if (!file) {
    alert("Choose a batch CSV first.");
    return;
  }

  setStatus("Running batch prediction");
  setBusy(true);
  const formData = new FormData();
  formData.append("file", file);
  if (currentThreshold !== null) {
    formData.append("threshold", currentThreshold);
  }

  const response = await fetch(`/api/predict/${encodeURIComponent(currentModelId)}/batch`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Batch prediction failed");
    setBusy(false);
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = getDownloadFilename(response, `${file.name.replace(/\.csv$/i, "")}_predictions.csv`);
  link.click();
  URL.revokeObjectURL(url);
  renderBatchPredictionStatus(link.download);
  document.querySelector("#predictionOutput").textContent = `Downloaded ${link.download}`;
  loadPredictionAuditLog();
  loadDeploymentReadiness(currentModelId);
  loadModelMonitoring(currentModelId);
  setStatus("Batch predictions downloaded");
  showToast("Batch prediction ready", `${link.download} has been downloaded.`);
  setBusy(false);
  showView("api");
  showDeployTab("results");
}

async function downloadModelReport() {
  setStatus("Exporting model report");
  const response = await fetch("/api/models/export");
  if (!response.ok) {
    alert(await readErrorMessage(response));
    setStatus("Report export failed");
    return;
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = getDownloadFilename(response, "automate-model-report.csv");
  link.click();
  URL.revokeObjectURL(url);
  setStatus("Model report exported");
  showToast("Report exported", `${link.download} has been downloaded.`);
}

async function downloadProjectSnapshot() {
  const [datasetsResponse, modelsResponse, samplesResponse] = await Promise.all([
    fetch("/api/datasets"),
    fetch("/api/models"),
    fetch("/api/sample-datasets"),
  ]);
  const snapshot = {
    exported_at: new Date().toISOString(),
    active_dataset_id: currentDatasetId,
    active_target: document.querySelector("#targetColumn").value || null,
    datasets: datasetsResponse.ok ? (await datasetsResponse.json()).datasets : [],
    models: modelsResponse.ok ? (await modelsResponse.json()).models : [],
    samples: samplesResponse.ok ? (await samplesResponse.json()).samples : [],
    last_dashboard: lastDashboardData,
  };
  const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `automate-project-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
  setStatus("Project snapshot saved");
}

async function initializeApp() {
  await loadSampleDatasets();
  await loadDatasetRegistry();
  await loadModelRegistry();
  await loadPredictionAuditLog();
  if (!currentDatasetId && datasetRegistry.length) {
    await selectDataset(datasetRegistry[0].dataset_id);
  }
}

document.querySelector("#uploadForm").addEventListener("submit", uploadCsv);
document.querySelector("#sqliteForm").addEventListener("submit", connectSqlite);
document.querySelector("#loadDashboard").addEventListener("click", loadDashboard);
document.querySelector("#refreshDashboard").addEventListener("click", loadDashboard);
document.querySelector("#trainForm").addEventListener("submit", trainModels);
document.querySelector("#accuracyBoosterButton").addEventListener("click", runAccuracyBooster);
document.querySelector("#predictButton").addEventListener("click", runPrediction);
document.querySelector("#batchPredictButton").addEventListener("click", runBatchPrediction);
document.querySelector("#runPlaygroundPrediction").addEventListener("click", runPlaygroundPrediction);
document.querySelector("#useSampleScenario").addEventListener("click", useSampleScenario);
document.querySelector("#resetWhatIfInputs").addEventListener("click", resetWhatIfInputs);
document.querySelector("#refreshAuditLog").addEventListener("click", loadPredictionAuditLog);
document.querySelector("#refreshReadiness").addEventListener("click", () => loadDeploymentReadiness());
document.querySelector("#runDriftCheck").addEventListener("click", runDatasetDriftCheck);
document.querySelector("#refreshMonitoring").addEventListener("click", () => loadModelMonitoring());
document.querySelector("#refreshMonitoringOps").addEventListener("click", () => loadModelMonitoring());
document.querySelector("#refreshChampionBoard").addEventListener("click", () => loadChampionChallenger());
document.querySelector("#promoteToProduction").addEventListener("click", runPromoteToProduction);
document.querySelector("#retrainFromDrift").addEventListener("click", runRetrainFromDrift);
document.querySelector("#refreshApiDocs").addEventListener("click", () => loadModelApiDocs());
document.querySelectorAll("[data-deploy-tab]").forEach((button) => {
  button.addEventListener("click", () => showDeployTab(button.dataset.deployTab));
});
document.querySelector("#commandRunDrift").addEventListener("click", () => {
  showDeployTab("drift");
  runDatasetDriftCheck();
});
document.querySelector("#commandPromote").addEventListener("click", () => {
  showDeployTab("promotion");
  runPromoteToProduction();
});
document.querySelector("#commandDocs").addEventListener("click", () => {
  showDeployTab("docs");
  loadModelApiDocs();
});
batchPredictFileInput.addEventListener("change", () => {
  const file = batchPredictFileInput.files[0];
  batchPredictFileName.textContent = file ? file.name : "No batch CSV selected";
});
document.querySelector("#targetColumn").addEventListener("change", () => {
  updateProblemTypeSuggestion();
  currentCleaningPlan = null;
  renderCleaningAssistant();
  loadCleaningPlan();
});
document.querySelector("#problemType").addEventListener("change", () => {
  updateActiveTargetNote();
  renderWorkspaceOverview();
  loadModelSuggestions();
});
document.querySelector("#selectAllFeatures").addEventListener("click", selectAllFeatures);
document.querySelector("#clearIdFeatures").addEventListener("click", removeIdLikeFeatures);
document.querySelector("#suggestModelsButton").addEventListener("click", loadModelSuggestions);
document.querySelector("#refreshModels").addEventListener("click", loadModelRegistry);
document.querySelector("#exportModelReport").addEventListener("click", downloadModelReport);
document.querySelector("#quickDashboard").addEventListener("click", quickDashboard);
document.querySelector("#quickTrain").addEventListener("click", quickTrain);
document.querySelector("#downloadProject").addEventListener("click", downloadProjectSnapshot);
document.querySelector("#workspaceDashboard").addEventListener("click", quickDashboard);
document.querySelector("#workspaceTrain").addEventListener("click", quickTrain);
document.querySelector("#workspaceExport").addEventListener("click", downloadProjectSnapshot);
document.addEventListener("click", (event) => {
  const target = event.target instanceof Element ? event.target : event.target?.parentElement;
  const inspectButton = target?.closest("[data-inspect-model]");
  if (inspectButton) {
    openModelDrawer(inspectButton.dataset.inspectModel);
  }
});
document.querySelector("#closeModelDrawer").addEventListener("click", closeModelDrawer);
document.querySelector("#drawerCloseModel").addEventListener("click", closeModelDrawer);
document.querySelector("#modelDrawerBackdrop").addEventListener("click", closeModelDrawer);
document.querySelector("#drawerDeployModel").addEventListener("click", openDrawerModelInDeploy);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModelDrawer();
  }
});
document.querySelector("#clearDashboardFilters").addEventListener("click", async () => {
  dashboardFilters = {};
  await loadDashboard();
});
document.querySelector("#dashboardDatasetSelect").addEventListener("change", async (event) => {
  await selectDataset(event.target.value);
  await loadDashboard();
});
document.querySelector("#modelDatasetSelect").addEventListener("change", async (event) => {
  await selectDataset(event.target.value);
  showView("model");
});
document.querySelector("#refreshModelContext").addEventListener("click", async () => {
  if (currentDatasetId) {
    await loadDatasetProfile(currentDatasetId);
  }
});

showDeployTab("playground");
updateDeployCommandBar();
initializeApp();
