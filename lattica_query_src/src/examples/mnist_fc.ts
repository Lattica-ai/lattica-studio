import { LatticaQueryClient, LatticaQueryClientLocalDev } from "../index";
import { getTensorShape, imageToTensor, tensorToImage } from "../tensorUtils";
import * as workerApi from "../workerApi";
import { displayImage, argmax } from "../dev_utils";

const RUN_LOCAL = true;

const jwt =
  "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiMWQyMWQzOTQtZTNiNS00NTdhLTk4MmMtMzc0M2FiNDA3YjQwIiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzM3NDYyMDU5LCJleHAiOjE3NDAwNTQwNTksInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.gk8bzjqaIUdDtEmn25fxyMIkO2Vt8Z_KCnXAumyslpUxF4-qhXobvnxUZ-uhmgw9NY4LAUcLxjEx-gLu-MU-VwprfZA5_f8UbMW-x39e3SeoXW-dg3PBm1mOlxR-1nLV3aq62kVZeAba58fmErmcvbaO8-qVcd1WZDNXGNb9P1GjcGlJqC-rBKN_NZ-SyZSIMZKWOksGeG9FB4313sQd9aayG3Aex00Xkdf2FjKqRq3dB5EjGr5QXTXE1bAqyJvKHIBsPFr660y2jKcJNMeDgmWWLU93Y17xdzLwhsonaEHGLeHRRmdC49wBScIlKBCF4YsUYDV6TOk436BLslP1qA";

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
  // console.time("Offline phase");
  // await latticaClient.init();
  // console.log("Client initialized.");
  // console.timeEnd("Offline phase");
  //
  // console.time("Online phase");
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

  // const ct = await latticaClient.encrypt(pt);
  // console.log("Encrypted.");
  // const ct_res = await latticaClient.apply_query(ct);
  // console.log("Query finished.");
  // const pt_res = await latticaClient.decrypt(ct_res);
  // console.log("Decrypted.");
  // console.timeEnd("Online phase");

  const pt_expected = await workerApi.apply_clear(client.jwt, client.beUrl, pt);
  console.log("Apply Clear finished");

  // console.log(
  //   "result shape:",
  //   getTensorShape(pt_res),
  //   "expected shape:",
  //   getTensorShape(pt_expected),
  // );
  // console.log("result:", pt_res);
  console.log("expected:", pt_expected);

  displayImage(
    imgData,
    `decoded prediction: ${argmax(pt_expected as number[])}`,
    300,
  );
})();
