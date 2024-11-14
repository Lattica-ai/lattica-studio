import axios from "axios";
import * as devUtils from "./dev_utils";
const {BE_URL} = require("./config");

const sendHttpRequest = async (
  jwt: string,
  requestName: string,
  reqParams: Record<string, any> = {},
): Promise<any> => {
  try {
    const options = {
      method: "POST",
      url: `${BE_URL}${requestName}`,
      data: reqParams,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${jwt}`,
      },
    };

    const response = await axios(options);

    if (response.data.error) {
      throw new Error(
          `FAILED ${requestName} with error: ${response.data.error}`,
          );
    }
    return response.data.result;
  } catch (error: any) {
    throw new Error(`FAILED ${requestName} with error: ${error.message}`);
  }
};

const uploadFile = async (
  data: Uint8Array,
  uploadUrl: string,
): Promise<void> => {
  // TODO: const blob = new Blob([data], { type: 'application/octet-stream' });
  try {
    const response = await axios.put(uploadUrl, data, {
      headers: {
        "Content-Type": "application/octet-stream",
      },
    });

    console.log("File uploaded successfully:", response.status);
  } catch (error) {
    console.error("Error uploading file:", error);
  }
};

export const uploadEk = async (
  jwt: string,
  ek: Uint8Array,
): Promise<void> => {
  console.log("Uploading evaluation key");
  const uploadUrl = await sendHttpRequest(jwt, "/token/get_pk_upload_url");
  console.log("Upload URL:", uploadUrl);
  await uploadFile(ek, uploadUrl);
  console.log("Evaluation key uploaded");

  // in local dev mode:
  // await devUtils.mockUploadFile("pks", ekFilePath, uploadUrl);
  // console.log("Evaluation key uploaded to localhost");

  return uploadUrl;
};
