import { Tensor } from "./utils";
import * as toolkit from "./toolkit_wrapper";
import axios from "axios";
import { BE_URL, MOCK_BE } from "./config";

type WorkerResponse = string | Record<string, string>;

const sendHttpRequest = async (
  jwt: string,
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
  try {
    const response = await axios.post(`${BE_URL}/do_action`, reqData, {
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${jwt}`,
      },
    });

    const responseJson = response.data;
    console.log(`${actionName}: ${responseJson.status}`);

    if (responseJson.status === "Running" && withPolling) {
      return resampleResult(jwt, responseJson.executionId);
    }

    if (responseJson.action_result) {
      if (responseJson.action_result.error) {
        throw new Error(responseJson.action_result.error); // worker error
      }
      return responseJson.action_result.result;
    }
  } catch (error) {
    throw new Error(
      `FAILED api/${actionName} with error: ${error.response.data.error}`,
    );
  }
};

const resampleResult = async (
  jwt: string,
  executionId: string,
): Promise<WorkerResponse> => {
  // Query the status of a previously triggered action.
  console.log(`Polling for executionId: ${executionId}`);

  if (!executionId) {
    // TODO: prevent worker from returning executionId = 0
    return null;
  }

  await new Promise((resolve) => setTimeout(resolve, 1000)); // Sleep for 1 second

  return await sendHttpRequest(jwt, "get_action_result", { executionId });
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
  const binaryString = String.fromCharCode.apply(null, uint8Array);
  return btoa(binaryString);
};

export const get_user_init_data = async (
  jwt: string,
): Promise<{ serializedContext: Uint8Array; serializedHomseq: Uint8Array }> => {
  if (MOCK_BE) {
    const mockApiContext = await fetch("mock_data/api_context");
    const mockClientHomseq = await fetch("mock_data/client_hom_seq");
    return {
      serializedContext: new Uint8Array(await mockApiContext.arrayBuffer()),
      serializedHomseq: new Uint8Array(await mockClientHomseq.arrayBuffer()),
    };
  }
  const response = (await sendHttpRequest(jwt, "get_user_init_data")) as Record<
    string,
    string
  >;
  return {
    serializedContext: base64ToBytes(response.context_proto_str),
    serializedHomseq: base64ToBytes(response.homseq_proto_str),
  };
};

export const preprocessEk = async (jwt: string): Promise<void> => {
  if (MOCK_BE) {
    return;
  }
  await sendHttpRequest(jwt, "preprocess_pk");
};

export const apply_hom_pipeline = async (
  jwt: string,
  serialized_ct: Uint8Array,
  block_index: number,
  return_new_state: boolean = false,
): Promise<Uint8Array> => {
  if (MOCK_BE) {
    const mockCtRes = await fetch("mock_data/ct_res");
    return new Uint8Array(await mockCtRes.arrayBuffer());
  }
  const response = await sendHttpRequest(jwt, "apply_hom_pipeline", {
    ct_proto_str: bytesToBase64(serialized_ct),
    block_index,
    return_new_state,
  });
  return base64ToBytes(response as string);
};

export const apply_clear = async (jwt: string, pt: Tensor) => {
  const response = await sendHttpRequest(jwt, "apply_clear", {
    pt_proto_str: bytesToBase64(toolkit.dumps_proto_tensor(pt)),
  });
  return toolkit.load_proto_tensor(base64ToBytes(response as string));
};
