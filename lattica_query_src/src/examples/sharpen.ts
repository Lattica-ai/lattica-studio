import { LatticaQueryClient, LatticaQueryClientLocalDev } from "../index";
import { getTensorShape, imageToTensor, tensorToImage } from "../tensorUtils";
import * as workerApi from "../workerApi";
import { displayImage, fetchLocalImageAsDataUrl } from "../dev_utils";

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

// cat: 150x150
const grayScale = false;
const imagePath = "./src/examples/data/cat_150x150.png";

(async () => {
  const image = await fetchLocalImageAsDataUrl(imagePath);

  console.time("Offline phase");
  await client.init();
  console.log("Client initialized.");
  console.timeEnd("Offline phase");

  console.time("Online phase");
  let pt;
  let imgData;
  if (typeof image === "string") {
    imgData = image;
    pt = await imageToTensor(image, grayScale, 100, 100);
    console.log("Image converted to tensor:", getTensorShape(pt));
  } else {
    imgData = await tensorToImage(image);
    pt = image;
  }

  const ct = await client.encrypt(pt);
  console.log("Encrypted.");
  console.time("Run query");
  const ct_res = await client.apply_query(ct);
  console.log("Query finished.");
  console.timeEnd("Run query");
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

  displayImage(imgData, "Original Image", 300);

  const imgExpected = await tensorToImage(pt_expected);
  displayImage(imgExpected, "Expected Image", 300);

  const imgDec = await tensorToImage(pt_res);
  displayImage(imgDec, "Decoded Image", 300);
})();
