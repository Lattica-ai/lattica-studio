import * as fs from "fs";
import axios from "axios";
import * as devUtils from "./dev_utils";

const sendHttpRequest = async (
  jwt: string,
  requestName: string,
  reqParams: Record<string, any> = {},
): Promise<any> => {
  try {
    const options = {
      method: "POST",
      url: `${process.env.BE_URL}${requestName}`,
      data: reqParams,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${jwt}`,
      },
    };

    const response = await axios(options);

    if (response.status === 200 && response.data && !response.data.error) {
      return response.data.result;
    }

    throw new Error(
      `FAILED ${requestName} with error: ${response.data?.error || "Unknown error"}`,
    );
  } catch (error: any) {
    throw new Error(`FAILED ${requestName} with error: ${error.message}`);
  }
};

const uploadFile = async (
  filePath: string,
  uploadUrl: string,
): Promise<void> => {
  try {
    const fileBuffer = fs.readFileSync(filePath);

    const response = await axios.put(uploadUrl, fileBuffer, {
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
  ekFilePath: string,
): Promise<void> => {
  console.log("Uploading evaluation key");
  const uploadUrl = await sendHttpRequest(jwt, "/token/get_pk_upload_url");
  console.log("Upload URL:", uploadUrl);
  await uploadFile(ekFilePath, uploadUrl);
  console.log("Evaluation key uploaded");
  // TODO: for DEV
  devUtils.mockUploadFile("pks", ekFilePath, uploadUrl);
  console.log("Evaluation key uploaded to localhost");
  return uploadUrl;
};
