const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const fs = require("fs");

let mainWindow = null;

const runBackendStep = (payload) => {
  const scriptPath = path.join(__dirname, "backend.py");
  const pythonCmd = process.env.PYTHON_CMD || "python";
  return new Promise((resolve) => {
    const proc = spawn(pythonCmd, ["-u", scriptPath], {
      cwd: path.join(__dirname, ".."),
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {
      const text = data.toString();
      stdout += text;
      text.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("PROGRESS ")) {
          const payload = line.replace("PROGRESS ", "").trim();
          try {
            const parsed = JSON.parse(payload);
            mainWindow?.webContents.send("backend:progress", parsed);
          } catch (err) {
            console.error(`[backend] progress parse failed: ${err.message}`);
          }
        }
      });
      console.log(`[backend] ${text}`);
    });

    proc.stderr.on("data", (data) => {
      const text = data.toString();
      stderr += text;
      console.error(`[backend] ${text}`);
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        resolve({ ok: false, error: stderr || "Backend failed to run." });
        return;
      }
      try {
        const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
        const lastJsonLine = [...lines].reverse().find((line) => line.trim().startsWith("{"));
        if (!lastJsonLine) {
          resolve({ ok: false, error: "Backend returned no JSON output." });
          return;
        }
        const result = JSON.parse(lastJsonLine);
        resolve(result);
      } catch (err) {
        resolve({ ok: false, error: `Invalid backend response: ${err.message}` });
      }
    });

    proc.stdin.write(JSON.stringify(payload));
    proc.stdin.end();
  });
};

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1280,
    minHeight: 800,
    resizable: true,
    maximizable: true,
    backgroundColor: "#0b0d10",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "index.html"));
}

app.whenReady().then(() => {
  createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

ipcMain.handle("backend:run-step", async (_event, payload) => {
  return await runBackendStep(payload);
});

ipcMain.handle("backend:export-report", async (_event, payload) => {
  if (!mainWindow) return { ok: false, error: "Window not ready." };
  const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
    title: "Export Componentry Report",
    defaultPath: "componentry_report.json",
    filters: [
      { name: "JSON", extensions: ["json"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (canceled || !filePath) return { ok: false, canceled: true };
  try {
    fs.writeFileSync(filePath, JSON.stringify(payload || {}, null, 2), "utf-8");
    return { ok: true, path: filePath };
  } catch (err) {
    return { ok: false, error: err.message };
  }
});

ipcMain.handle("backend:reset", async (_event) => {
  return await runBackendStep({ step: "reset" });
});

ipcMain.handle("backend:export-page", async (_event, payload) => {
  if (!mainWindow) return { ok: false, error: "Window not ready." };
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: "Choose export folder",
    properties: ["openDirectory", "createDirectory"],
  });
  if (canceled || !filePaths?.length) {
    return { ok: false, canceled: true };
  }
  const exportBaseDir = filePaths[0];
  return await runBackendStep({ ...payload, exportBaseDir });
});

ipcMain.handle("backend:select-images", async (_event) => {
  if (!mainWindow) return { ok: false, error: "Window not ready." };
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: "Select microscope images",
    properties: ["openFile", "multiSelections"],
    filters: [{ name: "Images", extensions: ["tif", "tiff", "png", "jpg", "jpeg"] }],
  });
  if (canceled || !filePaths?.length) {
    return { ok: false, canceled: true, paths: [] };
  }
  return { ok: true, paths: filePaths };
});