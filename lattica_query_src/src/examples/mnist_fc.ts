import {
  getDemoToken,
  LatticaQueryClient,
  LatticaQueryClientLocalDev,
} from "../index";
import { getTensorShape, imageToTensor, tensorToImage } from "../tensorUtils";
import * as workerApi from "../workerApi";
import { displayImage, argmax, fetchLocalImageAsDataUrl } from "../dev_utils";

const RUN_LOCAL = true;

const MODEL_ID = "sketchToNumber";

// number 4: 200x200
const grayScale = true;
const imagePath = "./src/examples/data/four_200x200.png";

(async () => {
  // Either use hardcoded JWT or create a new one (if the model visibility is PUBLIC)
  const jwt = "<JWT>";
  // const jwt = await getDemoToken(MODEL_ID, "http://localhost:3050");
  console.log("Created demo token for public model", MODEL_ID, jwt);

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
    pt = await imageToTensor(image, grayScale, 28, 28);
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
    `Decoded prediction: ${argmax(pt_res as number[])}`,
    300,
  );
})();
