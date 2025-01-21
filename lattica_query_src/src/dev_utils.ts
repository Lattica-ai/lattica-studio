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

export const fetchLocalFile = async (filePath: string): Promise<Uint8Array> => {
  const data = await fetch(filePath);
  return new Uint8Array(await data.arrayBuffer());
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

export function argmax(arr: number[]): number {
  if (arr.length === 0) return -1;

  let maxIndex = 0;
  let maxValue = arr[0];

  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxValue) {
      maxValue = arr[i];
      maxIndex = i;
    }
  }

  return maxIndex;
}
