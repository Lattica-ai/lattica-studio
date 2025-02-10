interface WorkerMessage {
  operation: string;
  data: any;
}

interface WorkerResponse {
  success: boolean;
  data?: any;
  error?: string;
  log: boolean;
  logtype: string;
  logmsg: string;
}

export class WasmWorkerManager {
  private worker?: Worker;
  private logCallback?: (type: string, message: string) => void;

  async initWorker() {
    return new Promise((resolve, reject) => {
      const workerUrl = import.meta.url
        ? new URL("../dist/wasm/wasm_worker.js", import.meta.url).href
        : "../dist/wasm/wasm_worker.js";
      this.worker = new Worker(workerUrl);

      const handleInitMessage = (event: MessageEvent) => {
        if (event.data === "loaded_wasm_module") {
          console.log("WasmWorker initialized successfully");
          resolve(undefined);
        } else {
          console.error("WasmWorker init error", event.data);
          reject(event.data);
        }
        this.worker!.removeEventListener('message', handleInitMessage);
      };

      this.worker.addEventListener('message', handleInitMessage);

      this.worker.addEventListener('message', (event: MessageEvent) => {

        const { log, logtype, logmsg } = event.data;
        if (log !== true || log === undefined) {
          return;
        }
        if (logtype === 'log') {
          console.log(`[log]: 123 ${logmsg}`);
        } else if (logtype === 'error') {
          console.error(`[error]: 123 ${logmsg}`);
        }

        if (this.logCallback) {
          this.logCallback(logtype, logmsg);
        }
      });
    });
  }

  async callWorker(operation: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error("Worker is not initialized"));
        return;
      }
      console.log("...debug inside callWorker");
      const message: WorkerMessage = { operation, data };

      // Listen for the worker's response
      const handleMessage = (event: MessageEvent) => {
        const response: WorkerResponse = event.data;

        if (response.log) {
          return;
        }

        if (response.success) {
          resolve(response.data);
        } else {
          reject(new Error("Worker error: " + (response.error || "Unexpected error response" + JSON.stringify(response))));
        }

        // Clean up the event listener
        this.worker!.removeEventListener("message", handleMessage);
      };

      // Add the message listener
      this.worker!.addEventListener("message", handleMessage);

      // Post the message to the worker
      this.worker!.postMessage(message);

      console.log("...debug done callWorker");

    });
  }

  terminate() {
    this.worker!.terminate();
  }

  setLogCallback(callback: (type: string, message: string) => void) {
    this.logCallback = callback;
  }
}

let initPromise: Promise<void> | null = null;
export let workerManagerSingleton: WasmWorkerManager = new WasmWorkerManager();
export async function initWasm() {
  if (initPromise) {
    return initPromise;
  }
  console.log("Starting WASM initialization...");
  initPromise = (async () => {
    await workerManagerSingleton.initWorker();
  })();
  return initPromise;
}
