import { Tensor } from "./tensorUtils";
import { load_proto_tensor, dumps_proto_tensor } from "./toolkitWrapper";
import { CLIENT_INFO } from "./clientVersionInfo";
import { workerManagerSingleton } from "./wasm/wasm_worker_manager";

type WorkerResponse = Uint8Array | null;

const sendHttpRequest = async (
  jwt: string,
  beUrl: string,
  actionName: string,
  actionParams: Record<string, any> | null = null,
  serializedData: Uint8Array | null = null,
  withPolling: boolean = true,
): Promise<WorkerResponse> => {
  const reqData = {
    api_call: {
      action: actionName,
      params: {
        ...(actionParams || {}),
        client_info: CLIENT_INFO,
      },
    },
  };

  const formData = new FormData();
  formData.append("metadata", JSON.stringify(reqData));

  if (serializedData) {
    formData.append(
      "file",
      new Blob([serializedData], { type: "application/octet-stream" }),
      "data.bin",
    );
  }

  const response = await fetch(`${beUrl}/api/do_action`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${jwt}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`FAILED api/${actionName} with error: ${errorText}`);
  }

  const contentType = response.headers.get("content-type");
  if (!contentType) {
    throw new Error("No content-type header received");
  }

  if (contentType.includes("application/octet-stream")) {
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
  } else if (contentType.includes("application/json")) {
    const responseJson = await response.json();
    console.log(`${actionName}: ${responseJson.status}`);

    if (responseJson.status === "RUNNING" && withPolling) {
      return resampleResult(jwt, beUrl, responseJson.executionId);
    }

    if (responseJson.status === "ERROR") {
      throw new Error(`Error in action ${actionName}: ${responseJson.error}`);
    }

    return null;
  } else {
    throw new Error(`Unexpected content type: ${contentType}`);
  }
};

const resampleResult = async (
  jwt: string,
  beUrl: string,
  executionId: string,
): Promise<WorkerResponse> => {
  // Query the status of a previously triggered action.
  console.log(`Polling for executionId: ${executionId}`);

  await new Promise((resolve) => setTimeout(resolve, 1000)); // Sleep for 1 second

  return await sendHttpRequest(jwt, beUrl, "get_action_result", {
    executionId,
  });
};

export const get_user_init_data = async (
  jwt: string,
  beUrl: string,
): Promise<Uint8Array> => {
  const response = await sendHttpRequest(jwt, beUrl, "get_user_init_data");
  if (!response) {
    throw new Error("Failed to get user init data");
  }
  return response;
};

export const preprocessEk = async (
  jwt: string,
  beUrl: string,
): Promise<void> => {
  await sendHttpRequest(jwt, beUrl, "preprocess_pk");
};

export const apply_hom_pipeline = async (
  jwt: string,
  beUrl: string,
  serialized_ct: Uint8Array,
  block_index: number,
  return_new_state: boolean = false,
): Promise<Uint8Array> => {
  workerManagerSingleton.logConsole("Sending encrypted data to server...");
  const response = await sendHttpRequest(
    jwt,
    beUrl,
    "apply_hom_pipeline",
    {
      block_index,
      return_new_state,
    },
    serialized_ct,
  );
  if (!response) {
    throw new Error("Failed to apply hom pipeline");
  }
  workerManagerSingleton.logConsole("Received encrypted result from server.");
  return response;
};

export const apply_clear = async (jwt: string, beUrl: string, pt: Tensor) => {
  const response = await sendHttpRequest(
    jwt,
    beUrl,
    "apply_clear",
    {},
    dumps_proto_tensor(pt),
  );
  if (!response) {
    throw new Error("Failed to apply clear");
  }
  return load_proto_tensor(response);
};
