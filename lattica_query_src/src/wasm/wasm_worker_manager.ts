interface WorkerMessage {
  operation: string;
  data: any;
}

interface WorkerResponse {
  success: boolean;
  data?: any;
  error?: string;
}

export class WasmWorkerManager {
  private worker?: Worker;

  async initWorker() {
    return new Promise((resolve, reject) => {
      this.worker = new Worker('src/wasm/wasm_worker.js');

      const handleInitMessage = (event: MessageEvent) => {
        if (event.data === "loaded_wasm_module") {
          console.log("WasmWorker initialized successfully");
          resolve(undefined);
        }
        else {
          console.error("WasmWorker init error", event.data);
          reject(event.data);
        }
        this.worker!.removeEventListener('message', handleInitMessage);
      };
      this.worker.addEventListener('message', handleInitMessage);
    });
  }

  async callWorker(operation: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const message: WorkerMessage = { operation, data };

      // Listen for the worker's response
      const handleMessage = (event: MessageEvent) => {
        const response: WorkerResponse = event.data;

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
}