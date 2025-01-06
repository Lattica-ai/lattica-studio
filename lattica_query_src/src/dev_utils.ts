import { imageToTensor, LatticaQueryClient } from "./index";
import { getTensorShape } from "./tensorUtils";
import * as workerApi from "./workerApi";
import { Tensor } from "./tensorUtils";

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

export async function exampleRunImage(
  latticaClient: LatticaQueryClient,
  image: string | Tensor,
) {
  console.time("Offline phase");
  await latticaClient.init();
  console.log("Client initialized.");
  console.timeEnd("Offline phase");

  console.time("Online phase");
  let pt;
  if (typeof image === "string") {
    pt = await imageToTensor(image, true, 255);
    console.log("Image converted to tensor.");
  } else {
    pt = image;
  }

  const ct = await latticaClient.encrypt(pt);
  console.log("Encrypted.");
  const ct_res = await latticaClient.apply_query(ct);
  console.log("Query finished.");
  const pt_res = await latticaClient.decrypt(ct_res);
  console.log("Decrypted.");
  console.timeEnd("Online phase");

  const pt_expected = await workerApi.apply_clear(
    latticaClient.jwt,
    latticaClient.beUrl,
    pt,
  );
  console.log("Apply Clear finished");

  console.log(
    "result shape:",
    getTensorShape(pt_res),
    "expected shape:",
    getTensorShape(pt_expected),
  );
  console.log("result:", pt_res);
  console.log("expected:", pt_expected);

  return pt_res;
}
