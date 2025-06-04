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
      try {
        const workerUrl = import.meta.url
          ? new URL('./wasm/wasm_worker.js', import.meta.url).href
          : './wasm/wasm_worker.js'; // Fallback for non-ESM environments or if import.meta.url is not available

        // Initialize Worker
        this.worker = new Worker(workerUrl);

        // Define onWorkerInit event handlers
        const onWorkerInitError = (event: ErrorEvent) => {
          console.error('WasmWorker internal error during initialization:', event.message, event);
          if (this.worker) {
            this.worker.removeEventListener('message', onWorkerInitMessage);
            this.worker.removeEventListener('error', onWorkerInitError);
          }
          reject(new Error(`WasmWorker internal error: ${event.message}`));
          event.preventDefault();
        };

        const onWorkerInitMessage = (event: MessageEvent) => {
          if (this.worker) {
            this.worker.removeEventListener('message', onWorkerInitMessage);
            this.worker.removeEventListener('error', onWorkerInitError);
          }
          if (event.data === 'loaded_wasm_module') {
            console.log('WasmWorker initialized successfully');
            resolve(undefined);
          } else {
            const errMsg = `WasmWorker unexpected message: ${String(event.data)}`;
            console.error(errMsg, event.data);
            reject(new Error(errMsg));
          }
        };

        this.worker.addEventListener('message', onWorkerInitMessage);
        this.worker.addEventListener('error', onWorkerInitError);

        this.worker.addEventListener('message', (event: MessageEvent) => {
          const { log, logtype, logmsg } = event.data;
          if (log !== true || log === undefined) {
            return;
          }

          if (this.logCallback) {
            this.logCallback(logtype, logmsg);
          }
        });
      } catch (error: any) {
        const errorMessage = 'Failed to instantiate WasmWorker';
        console.error(errorMessage, error);
        reject(new Error(error.message ? `${errorMessage}: ${error.message}` : errorMessage));
      }
    });
  }

  async callWorker(operation: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error('Worker is not initialized'));
        return;
      }
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
          reject(response.error);
        }

        // Clean up the event listener
        this.worker!.removeEventListener('message', handleMessage);
      };

      // Add the message listener
      this.worker!.addEventListener('message', handleMessage);

      // Post the message to the worker
      this.worker!.postMessage(message);
    });
  }

  terminate() {
    this.worker!.terminate();
  }

  setLogCallback(callback: (type: string, message: string) => void) {
    this.logCallback = callback;
  }

  logConsole(message: string) {
    this.logCallback?.('Console', message);
  }
}

let initPromise: Promise<void> | null = null;
export let workerManagerSingleton: WasmWorkerManager = new WasmWorkerManager();
export async function initWasm() {
  if (initPromise) {
    return initPromise;
  }
  console.log('Starting WASM initialization...');
  initPromise = (async () => {
    await workerManagerSingleton.initWorker();
  })();
  return initPromise;
}

export function setLogCallback(callback: (type: string, message: string) => void) {
  workerManagerSingleton.setLogCallback(callback);
}
