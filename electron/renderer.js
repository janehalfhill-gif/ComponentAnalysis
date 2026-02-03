const navItems = document.querySelectorAll(".nav__item");
const pages = document.querySelectorAll(".page");
const dockTabs = document.querySelectorAll(".dock-tab");
const dockPanels = document.querySelectorAll("[data-dock-panel]");
const outputDock = document.getElementById("output-dock");
const dockToggleBtn = document.getElementById("dockToggleBtn");
const fileInput = document.getElementById("imageInput");
const fileMeta = document.getElementById("fileMeta");
const fileList = document.getElementById("fileList");
const stepButtons = document.querySelectorAll("[data-action]");
const resultsOutput = document.getElementById("resultsOutput");
const imageCountEl = document.querySelector(".stat:nth-child(1) .stat__value");
const grainCountEl = document.querySelector(".stat:nth-child(2) .stat__value");
const clusterCountEl = document.querySelector(".stat:nth-child(3) .stat__value");
const progressBar = document.getElementById("progressBar");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const grainGallery = document.getElementById("grainGallery");
const embeddingGallery = document.getElementById("embeddingGallery");
const clusterGallery = document.getElementById("clusterGallery");
const resultsGallery = document.getElementById("resultsGallery");
const clusterBadge = document.getElementById("clusterBadge");
const componentryTable = document.getElementById("componentryTable");
const reviewClustersBtn = document.getElementById("reviewClustersBtn");
const componentryStepBtn = document.getElementById("componentryStepBtn");
const exportBtn = document.getElementById("exportBtn");
const resetBtn = document.getElementById("resetBtn");
const imageModal = document.getElementById("imageModal");
const modalImage = document.getElementById("modalImage");
const clusterReviewList = document.getElementById("clusterReviewList");
const clusterReviewBadge = document.getElementById("clusterReviewBadge");
const mergeClustersBtn = document.getElementById("mergeClustersBtn");
const splitClusterBtn = document.getElementById("splitClusterBtn");
const splitKInput = document.getElementById("splitKInput");
const clearClusterSelectionBtn = document.getElementById("clearClusterSelectionBtn");
const saveGrainsBtn = document.getElementById("saveGrainsBtn");
const saveEmbeddingsBtn = document.getElementById("saveEmbeddingsBtn");
const saveClustersBtn = document.getElementById("saveClustersBtn");
const saveResultsBtn = document.getElementById("saveResultsBtn");
const learningStatusEl = document.getElementById("learningStatus");
const embeddingsCachedEl = document.getElementById("embeddingsCached");
const diagImagesEl = document.getElementById("diagImages");
const diagGrainsEl = document.getElementById("diagGrains");
const diagClustersEl = document.getElementById("diagClusters");
const diagTimestampEl = document.getElementById("diagTimestamp");
const diagTrainedAtEl = document.getElementById("diagTrainedAt");
const diagTrainSamplesEl = document.getElementById("diagTrainSamples");
const rangeInputs = document.querySelectorAll('input[type="range"]');

const stepOrder = ["input", "step1", "step2", "step3", "step4"];
let selectedPaths = [];
let lastGrainSamples = [];
let lastClusterSamples = [];
let clustersReady = false;
let clustersReviewed = false;
let activePageId = "page-input";
let clusterAssignments = {};
let lastReport = {};
let selectedClusterIds = new Set();
let lastClassOptions = [];
let lastEmbedCached = null;
let newClusterIds = new Set();
let lastTrainingInfo = null;
let lastBackgroundClass = "background";

const setNavDot = (name, active) => {
  const dot = document.querySelector(`[data-dot="${name}"]`);
  if (!dot) return;
  dot.classList.toggle("is-active", Boolean(active));
};

const setCardEnabled = (stepId, enabled) => {
  const card = document.querySelector(`[data-step="${stepId}"]`);
  if (!card) return;
  card.classList.toggle("is-disabled", !enabled);
  card.querySelectorAll("input, select, button").forEach((el) => {
    if (stepId === "input" && el.type === "file") return;
    el.disabled = !enabled;
  });
  if (stepId === "step4" && componentryStepBtn && enabled && !clustersReviewed) {
    componentryStepBtn.disabled = true;
  }
};

const resetSteps = () => {
  stepOrder.forEach((stepId, index) => {
    setCardEnabled(stepId, stepId === "input");
    if (index > 0) {
      setCardEnabled(stepId, false);
    }
  });
  if (clusterBadge) clusterBadge.classList.remove("is-active");
  if (reviewClustersBtn) reviewClustersBtn.disabled = true;
  if (componentryStepBtn) componentryStepBtn.disabled = true;
  clustersReady = false;
  clustersReviewed = false;
  clusterAssignments = {};
  if (clusterReviewBadge) clusterReviewBadge.classList.remove("is-active");
  setNavDot("grains", false);
  setNavDot("embeddings", false);
  setNavDot("clusters", false);
  setNavDot("results", false);
};

const enableNextStep = (nextStep) => setCardEnabled(nextStep, true);

const updateStats = (data) => {
  if (data.images != null && imageCountEl) imageCountEl.textContent = data.images;
  if (data.grains != null && grainCountEl) grainCountEl.textContent = data.grains;
  if (data.clusters != null && clusterCountEl) clusterCountEl.textContent = data.clusters;
};

const setProgress = (percent) => {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));
  if (progressFill) progressFill.style.width = `${value}%`;
  if (progressText) progressText.textContent = `${value}%`;
};

const renderGallery = (container, paths, emptyText) => {
  if (!container) return;
  if (!Array.isArray(paths)) {
    paths = [];
  }
  container.innerHTML = "";
  if (!paths || paths.length === 0) {
    const empty = document.createElement("div");
    empty.className = "gallery__empty";
    empty.textContent = emptyText || "No samples yet.";
    container.appendChild(empty);
    return;
  }
  paths.forEach((path) => {
    const item = document.createElement("div");
    item.className = "gallery__item";
    const img = document.createElement("img");
    const normalized = path.replace(/\\/g, "/");
    img.src = `file:///${encodeURI(normalized)}`;
    img.dataset.full = img.src;
    img.alt = "Grain preview";
    item.appendChild(img);
    container.appendChild(item);
  });
};

const renderClusterGallery = (container, clusters, emptyText) => {
  if (!container) return;
  container.innerHTML = "";
  if (!clusters || clusters.length === 0) {
    const empty = document.createElement("div");
    empty.className = "gallery__empty";
    empty.textContent = emptyText || "No clusters yet.";
    container.appendChild(empty);
    return;
  }

  clusters.forEach((cluster) => {
    const item = document.createElement("div");
    item.className = "gallery__item gallery__item--labeled";

    const img = document.createElement("img");
    const normalized = (cluster.path || "").replace(/\\/g, "/");
    img.src = `file:///${encodeURI(normalized)}`;
    img.dataset.full = img.src;
    img.alt = "Cluster preview";

    const caption = document.createElement("div");
    caption.className = "gallery__caption";
    const clusterId = cluster.cluster_id ?? cluster.clusterId ?? cluster.id;
    const isNew = clusterId != null && newClusterIds.has(Number(clusterId));
    if (isNew) {
      caption.classList.add("gallery__caption--new");
    }
    caption.textContent = clusterId != null ? `Cluster ${clusterId}${isNew ? " • NEW" : ""}` : "Cluster";

    item.appendChild(img);
    item.appendChild(caption);
    container.appendChild(item);
  });
};

const renderComponentry = (percentages) => {
  if (!componentryTable) return;
  componentryTable.innerHTML = "";
  if (!percentages || Object.keys(percentages).length === 0) {
    const empty = document.createElement("div");
    empty.className = "results-table__empty";
    empty.textContent = "No results yet.";
    componentryTable.appendChild(empty);
    return;
  }
  Object.entries(percentages).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "results-table__row";
    const left = document.createElement("span");
    left.textContent = label;
    const right = document.createElement("span");
    right.textContent = `${value}%`;
    row.appendChild(left);
    row.appendChild(right);
    componentryTable.appendChild(row);
  });
};

const renderClusterReview = (clusters, classOptions) => {
  if (!clusterReviewList) return;
  clusterReviewList.innerHTML = "";
  lastClassOptions = classOptions || lastClassOptions;
  if (!clusters || clusters.length === 0) {
    const empty = document.createElement("div");
    empty.className = "review-empty";
    empty.textContent = "Run clustering to review clusters.";
    clusterReviewList.appendChild(empty);
    return;
  }
  const options = ["to choose", ...lastClassOptions];
  if (lastBackgroundClass && !options.includes(lastBackgroundClass)) {
    options.push(lastBackgroundClass);
  }
  clusters.forEach((item) => {
    const row = document.createElement("div");
    row.className = "review-row";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selectedClusterIds.has(item.cluster_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) {
        selectedClusterIds.add(item.cluster_id);
      } else {
        selectedClusterIds.delete(item.cluster_id);
      }
    });
    const img = document.createElement("img");
    const normalized = item.path.replace(/\\/g, "/");
    img.src = `file:///${encodeURI(normalized)}`;
    const label = document.createElement("div");
    const isNew = newClusterIds.has(Number(item.cluster_id));
    label.textContent = `Cluster ${item.cluster_id}${isNew ? " • NEW" : ""}`;
    const select = document.createElement("select");
    options.forEach((option) => {
      const opt = document.createElement("option");
      opt.value = option;
      opt.textContent = option;
      select.appendChild(opt);
    });
    const assigned = clusterAssignments[item.cluster_id];
    if (assigned && !options.includes(assigned)) {
      const opt = document.createElement("option");
      opt.value = assigned;
      opt.textContent = assigned;
      select.appendChild(opt);
    }
    select.value = assigned || "to choose";
    select.addEventListener("change", () => {
      clusterAssignments[item.cluster_id] = select.value;
    });
    row.appendChild(checkbox);
    row.appendChild(img);
    row.appendChild(label);
    row.appendChild(select);
    clusterReviewList.appendChild(row);
  });
};

const refreshClusterViews = (data) => {
  if (!data) return;
  updateStats(data || {});

  if (data.new_cluster_ids) {
    newClusterIds = new Set(data.new_cluster_ids.map((v) => Number(v)));
  } else {
    newClusterIds = new Set();
  }

  if (data.cluster_samples) {
    lastClusterSamples = data.cluster_samples;
    const clusterPaths = data.cluster_samples.map((item) => item.path);
    renderClusterGallery(clusterGallery, data.cluster_samples, "No clusters yet.");
    renderGallery(resultsGallery, clusterPaths, "No samples yet.");
    if (clusterBadge) clusterBadge.classList.add("is-active");
    if (reviewClustersBtn) reviewClustersBtn.disabled = false;
    if (clusterReviewBadge) clusterReviewBadge.classList.add("is-active");
    clusterAssignments = data.cluster_assignments || clusterAssignments;
    renderClusterReview(
      data.cluster_samples,
      data.class_labels || lastClassOptions || ["ash", "pumice", "crystalline", "lithic", "unknown"]
    );
    clustersReady = true;
    if (clustersReviewed && componentryStepBtn) {
      componentryStepBtn.disabled = false;
    }
    if (activePageId !== "page-clusters") setNavDot("clusters", true);
  }
};

const updateDiagnostics = () => {
  if (learningStatusEl) {
    learningStatusEl.textContent = lastTrainingInfo?.trained
      ? "Trained model available"
      : "No training (embeddings + clustering only)";
  }
  if (embeddingsCachedEl) {
    if (lastEmbedCached === null) {
      embeddingsCachedEl.textContent = "—";
    } else {
      embeddingsCachedEl.textContent = lastEmbedCached ? "Yes (cached)" : "No (recomputed)";
    }
  }
  if (diagImagesEl) diagImagesEl.textContent = lastReport?.stats?.images ?? "—";
  if (diagGrainsEl) diagGrainsEl.textContent = lastReport?.stats?.grains ?? "—";
  if (diagClustersEl) diagClustersEl.textContent = lastReport?.stats?.clusters ?? "—";
  if (diagTimestampEl) diagTimestampEl.textContent = lastReport?.timestamp ?? "—";
  if (diagTrainedAtEl) diagTrainedAtEl.textContent = lastTrainingInfo?.trained_at || "—";
  if (diagTrainSamplesEl) diagTrainSamplesEl.textContent = lastTrainingInfo?.samples ?? "—";
};

const readSettings = () => ({
  padding: Number(document.getElementById("paddingInput")?.value || 150),
  scale: Number(document.getElementById("scaleInput")?.value || 1.6),
  useParallel: Boolean(document.getElementById("parallelInput")?.checked),
  maxWorkers: Number(document.getElementById("workersInput")?.value || 8),
  batchSize: Number(document.getElementById("batchSizeInput")?.value || 32),
  seed: Number(document.getElementById("seedInput")?.value || 42),
  numClusters: Number(document.getElementById("clustersInput")?.value || 20),
  maxGrains: Number(document.getElementById("maxGrainsInput")?.value || 0),
  trainEpochs: Number(document.getElementById("trainEpochsInput")?.value || 5),
  trainBatchSize: Number(document.getElementById("trainBatchInput")?.value || 32),
  targetCategories: Number(document.getElementById("targetCategoriesInput")?.value || 0),
  backgroundClass: (document.getElementById("backgroundLabelInput")?.value || "background").trim(),
  ignoreBackground: Boolean(document.getElementById("ignoreBackgroundInput")?.checked),
  classes: (document.getElementById("classesInput")?.value || "")
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean),
  validation: {
    maxFiles: Number(document.getElementById("maxFilesInput")?.value || 200),
    maxFileSizeMb: Number(document.getElementById("maxFileSizeInput")?.value || 50),
    minDim: Number(document.getElementById("minDimInput")?.value || 200),
    minFocus: Number(document.getElementById("minFocusInput")?.value || 30),
    maxTotalPixelsMp: Number(document.getElementById("maxPixelsInput")?.value || 2000),
  },
});

const updateBackgroundClass = () => {
  const value = (document.getElementById("backgroundLabelInput")?.value || "background").trim();
  lastBackgroundClass = value || "background";
};

updateBackgroundClass();

navItems.forEach((item) => {
  item.addEventListener("click", () => {
    navItems.forEach((btn) => btn.classList.remove("nav__item--active"));
    item.classList.add("nav__item--active");
    const pageId = item.getAttribute("data-page");
    pages.forEach((page) => page.classList.remove("page--active"));
    const target = document.getElementById(pageId);
    if (target) target.classList.add("page--active");
    if (pageId) {
      activePageId = pageId;
      if (pageId === "page-grains") setNavDot("grains", false);
      if (pageId === "page-embeddings") setNavDot("embeddings", false);
      if (pageId === "page-clusters") setNavDot("clusters", false);
      if (pageId === "page-results") setNavDot("results", false);
    }
  });
});

dockTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    dockTabs.forEach((btn) => btn.classList.remove("dock-tab--active"));
    tab.classList.add("dock-tab--active");
    const target = tab.getAttribute("data-dock");
    dockPanels.forEach((panel) => {
      panel.classList.toggle("dock-panel--active", panel.dataset.dockPanel === target);
    });
    if (outputDock) {
      const showOverlay = target === "validation" || target === "diagnostics";
      outputDock.classList.toggle("is-expanded", showOverlay);
      outputDock.classList.toggle("is-overlay", showOverlay);
      outputDock.classList.remove("is-collapsed");
      if (dockToggleBtn) {
        dockToggleBtn.textContent = "Hide";
        dockToggleBtn.setAttribute("aria-expanded", "true");
      }
      document.body.classList.toggle("dock-overlay", showOverlay);
    }
    if (target === "validation") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  });
});

if (dockToggleBtn && outputDock) {
  dockToggleBtn.addEventListener("click", () => {
    if (outputDock.classList.contains("is-overlay")) return;
    const collapsed = outputDock.classList.toggle("is-collapsed");
    dockToggleBtn.textContent = collapsed ? "Show" : "Hide";
    dockToggleBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  });
}

stepButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const action = button.getAttribute("data-action");
    const nextStep = button.getAttribute("data-next");
    if (!action || !window.backend) return;
    if (["componentry", "train", "predict"].includes(action) && !clustersReviewed) {
      if (resultsOutput) {
        resultsOutput.textContent = "Review clusters before computing componentry.";
      }
      return;
    }

    button.disabled = true;
    const originalText = button.textContent;
    button.textContent = "Running...";
    if (progressBar) progressBar.classList.add("is-active");
    setProgress(0);
    if (resultsOutput) {
      resultsOutput.textContent = `Running ${action}...`;
    }

    const payload = {
      step: action,
      images: selectedPaths,
      settings: readSettings(),
      clusterAssignments,
    };

    try {
      const result = await window.backend.runStep(payload);
      if (!result.ok) {
        if (resultsOutput) {
          resultsOutput.textContent = result.error || "Something went wrong.";
        }
        return;
      }
      if (action === "embed") {
        lastEmbedCached = Boolean(result.data?.cached);
      }
      if (action === "train") {
        lastTrainingInfo = {
          trained: Boolean(result.data?.trained),
          trained_at: result.data?.trained_at || null,
          samples: result.data?.samples ?? null,
          classes: result.data?.class_labels || null,
          model_path: result.data?.model_path || null,
        };
      }
      if (resultsOutput) {
        const jsonText = JSON.stringify(result.data, null, 2);
        if (action === "embed" && result.data?.cached) {
          resultsOutput.textContent = `${jsonText}\n\nNote: embeddings loaded from cache (no model retraining).`;
        } else if (action === "train") {
          resultsOutput.textContent = `${jsonText}\n\nModel saved for future runs.`;
        } else {
          resultsOutput.textContent = jsonText;
        }
      }
      updateStats(result.data || {});

      if (result.data?.samples) {
        lastGrainSamples = result.data.samples;
        renderGallery(grainGallery, result.data.samples, "No grains yet.");
        renderGallery(embeddingGallery, result.data.samples, "No embeddings yet.");
        if (activePageId !== "page-grains") setNavDot("grains", true);
      }
      if (result.data?.cluster_samples) {
        lastClusterSamples = result.data.cluster_samples;
        const clusterPaths = result.data.cluster_samples.map((item) => item.path);
        refreshClusterViews(result.data);
      }
      if (result.data?.embeddings) {
        if (activePageId !== "page-embeddings") setNavDot("embeddings", true);
      }
      if (result.data?.classes) {
        renderComponentry(result.data.classes);
        if (activePageId !== "page-results") setNavDot("results", true);
      }
      lastReport = {
        stats: {
          images: imageCountEl ? Number(imageCountEl.textContent) : 0,
          grains: grainCountEl ? Number(grainCountEl.textContent) : 0,
          clusters: clusterCountEl ? Number(clusterCountEl.textContent) : 0,
        },
        componentry: result.data?.classes || null,
        settings: readSettings(),
        timestamp: new Date().toISOString(),
      };
      updateDiagnostics();

      if (nextStep) enableNextStep(nextStep);
    } catch (err) {
      if (resultsOutput) {
        resultsOutput.textContent = err?.message || "Failed to run step.";
      }
    } finally {
      if (progressBar) progressBar.classList.remove("is-active");
      setProgress(100);
      button.disabled = false;
      button.textContent = originalText;
    }
  });
});

resetSteps();
renderGallery(grainGallery, [], "No grains yet.");
renderGallery(embeddingGallery, [], "No embeddings yet.");
renderClusterGallery(clusterGallery, [], "No clusters yet.");
renderGallery(resultsGallery, [], "No samples yet.");
renderComponentry({});
if (reviewClustersBtn) reviewClustersBtn.disabled = true;
if (componentryStepBtn) componentryStepBtn.disabled = true;
renderClusterReview([], []);
updateDiagnostics();

const runClusterRefine = async (payload, runningText) => {
  if (!window.backend?.runStep) return;
  if (resultsOutput) resultsOutput.textContent = runningText || "Updating clusters...";
  try {
    const result = await window.backend.runStep(payload);
    if (!result.ok) {
      if (resultsOutput) resultsOutput.textContent = result.error || "Something went wrong.";
      return;
    }
    if (resultsOutput) resultsOutput.textContent = JSON.stringify(result.data, null, 2);
    // Clear selection after a successful refine.
    selectedClusterIds = new Set();
    refreshClusterViews(result.data || {});
    lastReport = {
      stats: {
        images: imageCountEl ? Number(imageCountEl.textContent) : 0,
        grains: grainCountEl ? Number(grainCountEl.textContent) : 0,
        clusters: clusterCountEl ? Number(clusterCountEl.textContent) : 0,
      },
      componentry: lastReport.componentry || null,
      settings: readSettings(),
      timestamp: new Date().toISOString(),
    };
    updateDiagnostics();
  } catch (err) {
    if (resultsOutput) resultsOutput.textContent = err?.message || "Failed to update clusters.";
  }
};

const runExportPage = async (pageId, label) => {
  if (!window.backend?.exportPage) return;
  if (resultsOutput) resultsOutput.textContent = `Saving ${label}...`;
  try {
    const result = await window.backend.exportPage({
      step: "export_page",
      page: pageId,
      settings: readSettings(),
      clusterAssignments,
    });
    if (!result.ok) {
      if (result.canceled) {
        if (resultsOutput) resultsOutput.textContent = "Save canceled.";
        return;
      }
      if (resultsOutput) resultsOutput.textContent = result.error || "Export failed.";
      return;
    }
    if (resultsOutput) resultsOutput.textContent = `Saved to ${result.data?.path || "export folder"}`;
  } catch (err) {
    if (resultsOutput) resultsOutput.textContent = err?.message || "Export failed.";
  }
};

if (clearClusterSelectionBtn) {
  clearClusterSelectionBtn.addEventListener("click", () => {
    selectedClusterIds = new Set();
    renderClusterReview(lastClusterSamples || [], lastClassOptions || []);
    if (resultsOutput) resultsOutput.textContent = "Cluster selection cleared.";
  });
}

if (mergeClustersBtn) {
  mergeClustersBtn.addEventListener("click", async () => {
    const ids = Array.from(selectedClusterIds || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
    if (ids.length < 2) {
      if (resultsOutput) resultsOutput.textContent = "Select 2+ clusters to merge.";
      return;
    }
    mergeClustersBtn.disabled = true;
    await runClusterRefine(
      {
        step: "merge",
        clustersToMerge: ids,
        settings: readSettings(),
        clusterAssignments,
      },
      `Merging clusters: ${ids.join(", ")}...`
    );
    mergeClustersBtn.disabled = false;
  });
}

if (splitClusterBtn) {
  splitClusterBtn.addEventListener("click", async () => {
    const ids = Array.from(selectedClusterIds || []).map((v) => Number(v)).filter((v) => Number.isFinite(v));
    if (ids.length !== 1) {
      if (resultsOutput) resultsOutput.textContent = "Select exactly 1 cluster to split.";
      return;
    }
    const k = Math.max(2, Math.min(10, Number(splitKInput?.value || 2)));
    splitClusterBtn.disabled = true;
    await runClusterRefine(
      {
        step: "split",
        clusterToSplit: ids[0],
        numSplits: k,
        settings: readSettings(),
        clusterAssignments,
      },
      `Splitting cluster ${ids[0]} into ${k} clusters...`
    );
    splitClusterBtn.disabled = false;
  });
}

if (saveGrainsBtn) {
  saveGrainsBtn.addEventListener("click", () => runExportPage("grains", "grains page"));
}
if (saveEmbeddingsBtn) {
  saveEmbeddingsBtn.addEventListener("click", () => runExportPage("embeddings", "embeddings page"));
}
if (saveClustersBtn) {
  saveClustersBtn.addEventListener("click", () => runExportPage("clusters", "clusters page"));
}
if (saveResultsBtn) {
  saveResultsBtn.addEventListener("click", () => runExportPage("results", "results page"));
}

if (exportBtn && window.backend?.exportReport) {
  exportBtn.addEventListener("click", async () => {
    const payload = lastReport && Object.keys(lastReport).length ? lastReport : { note: "No report yet." };
    const result = await window.backend.exportReport(payload);
    if (!result.ok && !result.canceled && resultsOutput) {
      resultsOutput.textContent = result.error || "Export failed.";
    }
    if (result.ok && resultsOutput) {
      resultsOutput.textContent = `Report saved to ${result.path}`;
    }
  });
}

if (resetBtn && window.backend?.resetState) {
  resetBtn.addEventListener("click", async () => {
    const res = await window.backend.resetState();
    if (resultsOutput) {
      resultsOutput.textContent = res.ok ? "Reset complete." : res.error || "Reset failed.";
    }
    resetSteps();
    selectedPaths = [];
    renderGallery(grainGallery, [], "No grains yet.");
    renderGallery(embeddingGallery, [], "No embeddings yet.");
    renderClusterGallery(clusterGallery, [], "No clusters yet.");
    renderGallery(resultsGallery, [], "No samples yet.");
    renderComponentry({});
    renderClusterReview([], []);
    if (fileMeta) fileMeta.textContent = "No files selected";
    if (fileList) fileList.innerHTML = "";
    if (fileInput) fileInput.value = "";
    setProgress(0);
    if (progressBar) progressBar.classList.remove("is-active");
    lastEmbedCached = null;
    newClusterIds = new Set();
    selectedClusterIds = new Set();
    lastReport = {};
    updateDiagnostics();
  });
}

if (reviewClustersBtn && componentryStepBtn) {
  reviewClustersBtn.addEventListener("click", () => {
    clustersReviewed = true;
    componentryStepBtn.disabled = !clustersReady;
    if (resultsOutput) {
      resultsOutput.textContent = "Cluster review complete. You can compute componentry.";
    }
    const clustersTab = document.querySelector('[data-page="page-clusters"]');
    clustersTab?.click();
  });
}

if (fileInput && fileMeta) {
  fileInput.addEventListener("change", () => {
    const files = fileInput.files ? Array.from(fileInput.files) : [];
    const count = files.length;
    fileMeta.textContent = count ? `${count} file(s) selected` : "No files selected";

    if (fileList) {
      fileList.innerHTML = "";
      files.slice(0, 5).forEach((file) => {
        const li = document.createElement("li");
        li.textContent = file.name;
        fileList.appendChild(li);
      });
      if (count > 5) {
        const li = document.createElement("li");
        li.textContent = `+ ${count - 5} more`;
        fileList.appendChild(li);
      }
    }

    if (count > 0) {
      selectedPaths = files
        .map((file) => file.path)
        .filter((value) => typeof value === "string" && value.length > 0);
      if (selectedPaths.length !== count && resultsOutput) {
        resultsOutput.textContent =
          "File paths are not available. Please select files via the Electron dialog.";
      }
      setCardEnabled("step1", true);
    } else {
      selectedPaths = [];
      stepOrder.slice(1).forEach((stepId) => setCardEnabled(stepId, false));
    }

    updateStats({ images: count });
    lastGrainSamples = [];
    lastClusterSamples = [];
    renderGallery(grainGallery, [], "No grains yet.");
    renderGallery(embeddingGallery, [], "No embeddings yet.");
    renderClusterGallery(clusterGallery, [], "No clusters yet.");
    renderGallery(resultsGallery, [], "No samples yet.");
    renderComponentry({});
    if (clusterBadge) clusterBadge.classList.remove("is-active");
    if (reviewClustersBtn) reviewClustersBtn.disabled = true;
    if (componentryStepBtn) componentryStepBtn.disabled = true;
    if (clusterReviewBadge) clusterReviewBadge.classList.remove("is-active");
    renderClusterReview([], []);
    setNavDot("grains", false);
    setNavDot("embeddings", false);
    setNavDot("clusters", false);
    setNavDot("results", false);
    clustersReady = false;
    clustersReviewed = false;
    clusterAssignments = {};
  });
}

if (window.backend?.onProgress) {
  window.backend.onProgress((data) => {
    if (!data || typeof data.percent === "undefined") return;
    if (progressBar) progressBar.classList.add("is-active");
    setProgress(data.percent);
    if (resultsOutput && data.message) {
      resultsOutput.textContent = data.message;
    }
    if (data.percent >= 100 && progressBar) {
      progressBar.classList.remove("is-active");
    }
  });
}

const updateRangeFill = (input) => {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const value = Number(input.value || 0);
  const percent = ((value - min) / (max - min)) * 100;
  input.style.background = `linear-gradient(90deg, #7db6ff 0%, #8ed6b6 60%, #f5d38b 100%) 0/ ${percent}% 100% no-repeat, #e7ecf5`;
};

rangeInputs.forEach((input) => {
  updateRangeFill(input);
  input.addEventListener("input", () => updateRangeFill(input));
});

let modalImages = [];
let modalIndex = -1;

const openModal = (src, images = [], index = 0) => {
  if (!imageModal || !modalImage) return;
  modalImages = Array.isArray(images) ? images : [];
  modalIndex = Number.isFinite(index) ? index : 0;
  modalImage.src = src;
  imageModal.classList.add("is-open");
  imageModal.setAttribute("aria-hidden", "false");
};

const showModalAtIndex = (index) => {
  if (!modalImage || modalImages.length === 0) return;
  const nextIndex = (index + modalImages.length) % modalImages.length;
  modalIndex = nextIndex;
  modalImage.src = modalImages[nextIndex];
};

const moveModal = (step) => {
  if (modalImages.length === 0) return;
  showModalAtIndex(modalIndex + step);
};

const closeModal = () => {
  if (!imageModal || !modalImage) return;
  imageModal.classList.remove("is-open");
  imageModal.setAttribute("aria-hidden", "true");
  modalImage.src = "";
  modalImages = [];
  modalIndex = -1;
};

document.addEventListener("click", (event) => {
  const target = event.target;
  if (target instanceof HTMLImageElement && target.dataset.full) {
    const gallery = target.closest(".gallery");
    const images = gallery
      ? Array.from(gallery.querySelectorAll("img"))
          .map((img) => img.dataset.full || img.src)
          .filter(Boolean)
      : [target.dataset.full];
    const index = images.indexOf(target.dataset.full);
    openModal(target.dataset.full, images, index >= 0 ? index : 0);
    return;
  }
  if (target instanceof HTMLElement && target.hasAttribute("data-modal-close")) {
    closeModal();
  }
});

document.addEventListener("keydown", (event) => {
  if (!imageModal || !imageModal.classList.contains("is-open")) return;
  if (event.key === "ArrowRight") {
    event.preventDefault();
    moveModal(1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    moveModal(-1);
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeModal();
  }
});

// Update range input values
const paddingInput = document.getElementById("paddingInput");
const scaleInput = document.getElementById("scaleInput");
const paddingValue = document.getElementById("paddingValue");
const scaleValue = document.getElementById("scaleValue");

if (paddingInput && paddingValue) {
  paddingInput.addEventListener("input", () => {
    paddingValue.textContent = paddingInput.value;
  });
}

if (scaleInput && scaleValue) {
  scaleInput.addEventListener("input", () => {
    scaleValue.textContent = scaleInput.value;
  });
}

const backgroundInput = document.getElementById("backgroundLabelInput");
if (backgroundInput) {
  backgroundInput.addEventListener("input", () => {
    updateBackgroundClass();
  });
}
