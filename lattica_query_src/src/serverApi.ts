const sendHttpRequest = async (
  jwt: string,
  beUrl: string,
  requestName: string,
  reqParams: Record<string, any> = {},
): Promise<any> => {
  const response = await fetch(`${beUrl}${requestName}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${jwt}`,
    },
    body: JSON.stringify(reqParams),
  });

  if (response.ok) {
    const responseJson = await response.json();

    if (!responseJson.error) {
      return responseJson.result;
    }
  }

  const errorText = await response.text();
  throw new Error(`FAILED ${requestName} with error: ${errorText}`);
};

const uploadFile = async (
  data: Uint8Array,
  uploadUrl: string,
): Promise<void> => {
  const blob = new Blob([data], {type: "application/octet-stream"});

  const response = await fetch(uploadUrl, {
    method: "PUT",
    body: blob,
    headers: {
      "Content-Type": "application/octet-stream",
    },
  });

  if (!response.ok) {
    throw new Error(`Error uploading file: ${response.status} ${response.statusText}`);
  }

  console.log("File uploaded successfully");
};

const alertUploadComplete = async (jwt: string, beUrl: string, s3Key: string): Promise<void> => {
    console.log("Alerting upload completion");
    await sendHttpRequest(jwt, beUrl, "/api/files/upload_complete", { s3Key });
    console.log("Upload completion alerted");
}

export const uploadEk = async (jwt: string, beUrl: string, ek: Uint8Array): Promise<string> => {
  console.log("Uploading evaluation key");
  const {s3Url, s3Key} = await sendHttpRequest(jwt, beUrl, "/api/token/get_pk_upload_url");
  console.log("Upload URL:", s3Url);
  console.log(s3Key);
  await uploadFile(ek, s3Url);
  console.log("Evaluation key uploaded");
  await alertUploadComplete(jwt, beUrl, s3Key);

  return s3Key;
};
