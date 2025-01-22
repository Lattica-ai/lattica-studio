import { LatticaQueryClient, LatticaQueryClientLocalDev } from "../index";
import { getTensorShape, imageToTensor, tensorToImage } from "../tensorUtils";
import * as workerApi from "../workerApi";
import { displayImage, argmax } from "../dev_utils";

const RUN_LOCAL = true;
const jwt = "<JWT>";

let client;
if (RUN_LOCAL) {
  client = new LatticaQueryClientLocalDev(jwt);
  console.log("DEV client created");
} else {
  client = new LatticaQueryClient(
    jwt,
    "https://api-alb-509680522.eu-central-1.elb.amazonaws.com", // old-dev env
  );
  console.log("Client created");
}

// number 4 (1,28,28)
const grayScale = true;
const image =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAA0ElEQVR4nGNgGAqA9+T/UpySs/7+xSUp1PDz73cnNMEYAQgd9/fvVy80uchfsyFqvv79ux1NLvTu3yviQFr4DNBCXjTJ13//1oJo079/H/KjSqm2//r7UBrIMLz192MYpuMfGoIY1n//5qE7/8ffv8+PzHByctrw928IuuStvwiwFV1SseHk3fNPTp78+PfvQRt0SSCQ49JgYDjy94sDFjkw4DmLaSgcePz9645Tcsbfm9K45PRf/N2BU6P/37+OOCV9/9ySwymp+aQSpxwVAQDGtGAPKDnpyQAAAABJRU5ErkJggg==";

(async () => {
  console.time("Offline phase");
  await client.init();
  console.log("Client initialized.");
  console.timeEnd("Offline phase");

  console.time("Online phase");
  let pt;
  let imgData;
  if (typeof image === "string") {
    imgData = image;
    pt = await imageToTensor(image, grayScale);
    console.log("Image converted to tensor:", getTensorShape(pt));
  } else {
    imgData = await tensorToImage(image);
    pt = image;
  }

  const ct = await client.encrypt(pt);
  console.log("Encrypted.");
  const ct_res = await client.apply_query(ct);
  console.log("Query finished.");
  const pt_res = await client.decrypt(ct_res);
  console.log("Decrypted.");
  console.timeEnd("Online phase");

  const pt_expected = await workerApi.apply_clear(client.jwt, client.beUrl, pt);
  console.log("Apply Clear finished");

  console.log(
    "result shape:",
    getTensorShape(pt_res),
    "expected shape:",
    getTensorShape(pt_expected),
  );
  console.log("result:", pt_res);
  console.log("expected:", pt_expected);

  displayImage(
    imgData,
    `decoded prediction: ${argmax(pt_res as number[])}`,
    300,
  );
})();
