import {
  LatticaQueryClient,
  LatticaQueryClientLocalDev,
  getDemoToken,
} from "../index";
import { imageToTensor, tensorToImage } from "../tensorUtils";
import * as workerApi from "../workerApi";
import { displayImage, argmax1D, fetchLocalImageAsDataUrl } from "../dev_utils";
import { workerManagerSingleton } from "../wasm/wasm_worker_manager";

const RUN_LOCAL = true;

let jwt = "<JWT>";
const MODEL_ID = "<ModelId>";

const MODEL_INPUT_SIZE = 32;

const grayScale = false;
const imagePath = "./src/examples/data/Truck_400x400.png";

document.addEventListener("DOMContentLoaded", () => {
  const logBox = document.getElementById("logBox");

  workerManagerSingleton.setLogCallback((type: string, message: string) => {
    if (!logBox) return;
    const logEntry = document.createElement("div");
    const isError = type === "WASM Error";
    if (isError) {
      logEntry.style.color = "red";
      logEntry.style.fontWeight = "bold";
    } else {
      if (type === "Console") {
        logEntry.style.fontWeight = "bold";
      }
    }
    logEntry.textContent = `[${type}] ${message}`;
    logBox.appendChild(logEntry);
    logBox.scrollTop = logBox.scrollHeight;
  });
});

(async () => {
  // Either use hardcoded JWT or create a new one (if the model visibility is PUBLIC)
  // jwt = await getDemoToken(MODEL_ID, "http://localhost:3050");
  // console.log("Created demo token for public model", MODEL_ID, jwt);

  let client;
  if (RUN_LOCAL) {
    client = new LatticaQueryClientLocalDev(jwt);
    workerManagerSingleton.logConsole("DEV client created");
  } else {
    client = new LatticaQueryClient(
      jwt,
      "https://gini.api.b543.lattica.dev", // Gini
    );
    workerManagerSingleton.logConsole("Client created");
  }

  const image = await fetchLocalImageAsDataUrl(imagePath);

  console.time("Offline phase");
  await client.init();
  workerManagerSingleton.logConsole("Client initialized.");
  console.timeEnd("Offline phase");

  console.time("Online phase");
  let pt;
  let imgData;
  if (typeof image === "string") {
    imgData = image;
    pt = await imageToTensor(
      image,
      grayScale,
      true,
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE,
    );
    workerManagerSingleton.logConsole(
      `Image converted to tensor: (${pt.shape})`,
    );
  } else {
    imgData = await tensorToImage(image, true);
    pt = image;
  }

  const pt_res = await client.runQuery(pt);
  console.timeEnd("Online phase");

  const pt_expected = await workerApi.apply_clear(client.jwt, client.beUrl, pt);
  workerManagerSingleton.logConsole("Apply Clear finished");

  workerManagerSingleton.logConsole(
    ["result shape:", pt_res.shape, "expected shape:", pt_expected.shape].join(
      " ",
    ),
  );
  console.log("result:", pt_res);
  console.log("expected:", pt_expected);

  const modelMetadataBinary = client.getModelMetadata();
  const jsonString = new TextDecoder().decode(modelMetadataBinary);
  const modelMetadata = JSON.parse(jsonString);

  displayImage(
    await tensorToImage(pt, true),
    `Prediction: ${modelMetadata.labels[argmax1D(pt_res)]} (actual: ${modelMetadata.labels[argmax1D(pt_expected)]})`,
    300,
  );

  displayImage(image, "Original Image", 300);
})();
