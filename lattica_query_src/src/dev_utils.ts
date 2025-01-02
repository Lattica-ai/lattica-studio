export const triggerEvkCopy = async (
  jwt: string,
  beUrl: string,
  ek: Uint8Array,
  s3Key: string,
): Promise<void> => {
  console.log("Triggering EVK copy");

  const fileName = s3Key.split("/").pop();
  console.log("EVK name:", fileName);

  const formData = new FormData();
  const blob = new Blob([ek], { type: "application/octet-stream" });
  formData.append(fileName, blob);

  const response = await fetch(`${beUrl}/files/dev_save_ek_local`, {
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
