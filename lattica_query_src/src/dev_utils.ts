import { CLIENT_INFO } from "./clientVersionInfo";
import { Tensor } from "./tensorUtils";

export const triggerEvkCopy = async (
  jwt: string,
  beUrl: string,
  ek: Uint8Array,
  s3Key: string,
): Promise<void> => {
  console.log("Triggering EVK copy");

  const fileName = s3Key.split("/").pop();
  if (!fileName) throw new Error("Invalid S3 key");
  console.log("EVK name:", fileName);

  const formData = new FormData();
  const blob = new Blob([ek], { type: "application/octet-stream" });
  formData.append(fileName, blob);

  // Add client version information
  formData.append("client_module", CLIENT_INFO.module);
  formData.append("client_version", CLIENT_INFO.version);

  const response = await fetch(`${beUrl}/api/files/dev_save_ek_local`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${jwt}`,
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(
      `Error triggering EVK copy: ${response.status} ${response.statusText}`,
    );
  }
};

export const fetchLocalBinaryFile = async (
  filePath: string,
): Promise<Uint8Array> => {
  const data = await fetch(filePath);
  return new Uint8Array(await data.arrayBuffer());
};

export const fetchLocalImageAsDataUrl = async (
  filePath: string,
): Promise<string> => {
  const response = await fetch(filePath);

  const blob = await response.blob();

  // Convert Blob to Base64 Data URL
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Error reading blob as Data URL."));
    reader.readAsDataURL(blob);
  });
};

export function displayImage(data: string, title?: string, width?: number) {
  const container = document.createElement("div"); // Create a container

  if (title) {
    const titleElement = document.createElement("h3"); // Create a title element
    titleElement.textContent = title;
    container.appendChild(titleElement); // Add the title to the container
  }

  const img = new Image();
  img.src = data;
  if (width) img.width = width;

  container.appendChild(img); // Add the image to the container
  document.body.appendChild(container); // Add the container to the body
}

export function argmax1D(arr: Tensor): number {
  if (arr.shape.length !== 1) {
    throw new Error("Input must be a 1D array.");
  }
  const len = arr.shape[0];
  let maxVal = arr.get(0);
  let maxIdx = 0;

  for (let i = 1; i < len; i++) {
    const val = arr.get(i);
    if (val > maxVal) {
      maxVal = val;
      maxIdx = i;
    }
  }
  return maxIdx;
}
