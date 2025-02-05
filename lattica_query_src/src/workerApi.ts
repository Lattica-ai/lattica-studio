import { Tensor } from "./tensorUtils";
import * as toolkit from "./toolkitWrapper";

type WorkerResponse = string | Record<string, string>;

const sendHttpRequest = async (
  jwt: string,
  beUrl: string,
  actionName: string,
  actionParams: Record<string, any> | null = null,
  withPolling: boolean = true,
): Promise<WorkerResponse> => {
  const reqData = {
    api_call: {
      action: actionName,
      params: actionParams || {},
    },
  };

  const response = await fetch(`${beUrl}/api/do_action`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${jwt}`,
    },
    body: JSON.stringify(reqData),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`FAILED api/${actionName} with error: ${errorText}`);
  }
  const responseJson = await response.json();
  console.log(`${actionName}: ${responseJson.status}`);

  if (responseJson.status === "RUNNING" && withPolling) {
    return resampleResult(jwt, beUrl, responseJson.executionId);
  }

  if (responseJson.action_result) {
    if (responseJson.action_result.error) {
      throw new Error(responseJson.action_result.error); // worker error
    }
    return responseJson.action_result.result;
  }
};

const resampleResult = async (
  jwt: string,
  beUrl: string,
  executionId: string,
): Promise<WorkerResponse> => {
  // Query the status of a previously triggered action.
  console.log(`Polling for executionId: ${executionId}`);

  if (!executionId) {
    // TODO: prevent worker from returning executionId = 0
    return "";
  }

  await new Promise((resolve) => setTimeout(resolve, 1000)); // Sleep for 1 second

  return await sendHttpRequest(jwt, beUrl, "get_action_result", {
    executionId,
  });
};

const base64ToBytes = (base64: string): Uint8Array => {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
};

const bytesToBase64 = (uint8Array: Uint8Array): string => {
  const chunkSize = 0x8000; // Process 32 KB chunks to avoid call stack issues
  let binaryString = "";

  for (let i = 0; i < uint8Array.length; i += chunkSize) {
    const chunk = uint8Array.subarray(i, i + chunkSize);
    binaryString += String.fromCharCode.apply(null, Array.from(chunk));
  }
  return btoa(binaryString);
};

export const get_user_init_data = async (
  jwt: string,
  beUrl: string,
): Promise<{ serializedContext: Uint8Array; serializedHomseq: Uint8Array }> => {
  const response = (await sendHttpRequest(
    jwt,
    beUrl,
    "get_user_init_data",
  )) as Record<string, string>;
  return {
    serializedContext: base64ToBytes(response.context_proto_str),
    serializedHomseq: base64ToBytes(response.homseq_proto_str),
  };
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
  const response = await sendHttpRequest(jwt, beUrl, "apply_hom_pipeline", {
    ct_proto_str: bytesToBase64(serialized_ct),
    block_index,
    return_new_state,
  });
  return base64ToBytes(response as string);
};

export const apply_clear = async (jwt: string, beUrl: string, pt: Tensor) => {
  const response = await sendHttpRequest(jwt, beUrl, "apply_clear", {
    pt_proto_str: bytesToBase64(toolkit.dumps_proto_tensor(pt)),
  });
  return toolkit.load_proto_tensor(base64ToBytes(response as string));
};
