import { getTensorShape, tensorToImage, imageToTensor } from "../tensorUtils";
import { displayImage, fetchLocalImageAsDataUrl } from "../dev_utils";

// number 4: 28x28
// const grayScale = true;
// const imagePath = "./src/examples/data/four_28x28.png";

// frog: 600x400
const grayScale = false;
const imagePath = "./src/examples/data/frog_600x400.JPEG";

(async () => {
  const imageData = await fetchLocalImageAsDataUrl(imagePath);

  console.log("original image base64 length:", imageData.length);
  displayImage(imageData, "original");

  const tensor = await imageToTensor(imageData, grayScale);
  console.log("tensor shape:", getTensorShape(tensor));
  console.log(tensor);

  const imgDataRec = await tensorToImage(tensor, 255);
  console.log("reconstructed image base64 length:", imgDataRec.length);
  displayImage(imgDataRec, "reconstructed");

  const tensorRec = await imageToTensor(imgDataRec, grayScale);
  console.log("reconstructed tensor shape:", getTensorShape(tensorRec));
  console.log(tensorRec);
})();
