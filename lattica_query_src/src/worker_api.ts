import axios from "axios";
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
    const response = await axios.post(
      `${process.env.BE_URL}/do_action`,
      reqData,
      {
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${jwt}`,
        },
      },
    );

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

const base64ToBuffer = (base64: string): Buffer => {
  return Buffer.from(base64, "base64");
};

export const get_user_init_data = async (
  jwt: string,
): Promise<{ serializedContext: Buffer; serializedHomseq: Buffer }> => {
  const response = await sendHttpRequest(jwt, "get_user_init_data");
  return {
    serializedContext: base64ToBuffer(response.context_proto_str),
    serializedHomseq: base64ToBuffer(response.homseq_proto_str),
  };
};

export const preprocessEk = async (jwt: string): Promise<void> => {
  await sendHttpRequest(jwt, "preprocess_pk");
};

export const apply_hom_pipeline = async (
  jwt: string,
  serialized_ct: Buffer,
  block_index: number,
  return_new_state: boolean = false,
): Promise<Buffer> => {
  const response = await sendHttpRequest(jwt, "apply_hom_pipeline", {
    ct_proto_str: serialized_ct.toString("base64"),
    block_index,
    return_new_state,
  });
  return base64ToBuffer(response);
};
