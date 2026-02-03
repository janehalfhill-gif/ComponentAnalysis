const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("backend", {
  runStep: (payload) => ipcRenderer.invoke("backend:run-step", payload),
  onProgress: (handler) => ipcRenderer.on("backend:progress", (_event, data) => handler(data)),
  exportReport: (payload) => ipcRenderer.invoke("backend:export-report", payload),
  exportPage: (payload) => ipcRenderer.invoke("backend:export-page", payload),
  resetState: () => ipcRenderer.invoke("backend:reset"),
});
