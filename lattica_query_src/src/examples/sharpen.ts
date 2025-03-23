import {
  LatticaQueryClient,
  LatticaQueryClientLocalDev,
  getDemoToken,
} from "../index";
import { getTensorShape, imageToTensor, tensorToImage } from "../tensorUtils";
import * as workerApi from "../workerApi";
import { displayImage, fetchLocalImageAsDataUrl } from "../dev_utils";
import { workerManagerSingleton } from "../wasm/wasm_worker_manager";

const RUN_LOCAL = true;

const MODEL_ID = "imageEnhancement";

const MODEL_INPUT_SIZE = 200;

// cat: 150x150
const grayScale = false;
const imagePath = "./src/examples/data/cat_150x150.png";

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
  const jwt = "<JWT>";
  // const jwt = await getDemoToken(MODEL_ID, "http://localhost:3050");
  console.log("Created demo token for public model", MODEL_ID, jwt);

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
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE,
    );
    workerManagerSingleton.logConsole(
      ["Image converted to tensor:", getTensorShape(pt)].join(""),
    );
  } else {
    imgData = await tensorToImage(image);
    pt = image;
  }

  //================ using runInferenceDemo
  // const ct = await client.encrypt(pt);
  // workerManagerSingleton.logConsole("Encrypted.");
  // console.time("Run query");
  // const ct_res = await client.apply_query(ct);
  // workerManagerSingleton.logConsole("Query finished.");
  // console.timeEnd("Run query");
  // const pt_res = await client.decrypt(ct_res);
  // workerManagerSingleton.logConsole("Decrypted.");
  // console.timeEnd("Online phase");

  //================ using runInference
  const pt_res = await client.runQuery(pt);

  const pt_expected = await workerApi.apply_clear(client.jwt, client.beUrl, pt);
  workerManagerSingleton.logConsole("Apply Clear finished");

  workerManagerSingleton.logConsole(
    [
      "result shape:",
      getTensorShape(pt_res),
      "expected shape:",
      getTensorShape(pt_expected),
    ].join(" "),
  );
  console.log("result:", pt_res);
  console.log("expected:", pt_expected);

  displayImage(imgData, "Original Image", 300);

  const imgDec = await tensorToImage(pt_res);
  displayImage(imgDec, "Decoded Image", 300);

  const imgExpected = await tensorToImage(pt_expected);
  displayImage(imgExpected, "Expected Image", 300);
})();
