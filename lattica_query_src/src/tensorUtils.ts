import { DataType } from "./serialization/generic";

export type Tensor = number | Tensor[];

export const getTensorShape = (tensor: Tensor): number[] => {
  if (typeof tensor === "number") {
    return [];
  }
  return [tensor.length, ...getTensorShape(tensor[0])];
};

export const getTensorStrides = (shape: number[]): number[] => {
  const strides = Array(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
};

export const flattenTensor = (tensor: Tensor): number[] => {
  if (typeof tensor === "number") {
    return [tensor]; // Base case: a single number.
  }
  return tensor.reduce<number[]>(
    (acc, value) => acc.concat(flattenTensor(value)),
    [],
  );
};

export const dataToTensor = (data: Uint8Array, dtype: DataType) => {
  const buffer = data.slice().buffer; // Copy the data to a new buffer
  switch (dtype) {
    case DataType.DOUBLE:
      return new Float64Array(buffer);
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
};

export const reshapeWithStrides = (
  flatArray: Float64Array,
  shape: number[],
  strides: number[],
) => {
  // Create a function to access the elements in the flattened array using strides
  function getElement(indices: number[]): number {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * strides[i];
    }
    return flatArray[index];
  }

  // Recursively create the multi-dimensional array based on shape and strides
  function createMultiDimensionalArray(
    dimIndex = 0,
    indices: number[] = [],
  ): Tensor {
    if (dimIndex === shape.length) {
      return getElement(indices);
    }

    const result = [];
    for (let i = 0; i < shape[dimIndex]; i++) {
      indices[dimIndex] = i;
      result.push(createMultiDimensionalArray(dimIndex + 1, indices));
    }
    return result;
  }

  return createMultiDimensionalArray();
};

/**
 * Converts a Base64 image data URL to an RGBA matrix.
 * @param base64 - The Base64 image data URL.
 * @param grayScale - Whether the image is in grayscale.
 * @param scaling - The scaling factor to divide the pixel values by.
 * @returns A promise resolving to a 3D array representing image in (C,H,W) format
 */
export async function imageToTensor(
  base64: string,
  grayScale: boolean = false,
  scaling: number = 255,
): Promise<Tensor> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = base64;

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width;
      canvas.height = img.height;

      const context = canvas.getContext("2d");
      if (!context) {
        reject(new Error("Failed to get canvas context."));
        return;
      }

      // Draw the image onto the canvas.
      context.drawImage(img, 0, 0);

      // Extract image data.
      const imageData = context.getImageData(0, 0, img.width, img.height);
      const { width, height, data } = imageData;

      const channels = grayScale ? 1 : 3;
      const tensor: number[][][] = Array.from({ length: channels }, () =>
        Array.from({ length: height }, () => Array(width).fill(0)),
      );

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4; // RGBA data index
          if (grayScale) {
            tensor[0][y][x] = data[idx] / scaling; // just take the first channel
          } else {
            tensor[0][y][x] = data[idx] / scaling;
            tensor[1][y][x] = data[idx + 1] / scaling;
            tensor[2][y][x] = data[idx + 2] / scaling;
          }
        }
      }

      resolve(tensor);
    };

    img.onerror = (err) => reject(new Error(`Failed to load image: ${err}`));
  });
}

/**
 * Converts a 3D array representing image in (C,H,W) format, to a Base64 image data URL.
 * @param tensor - The 3D array representing the image in (C,H,W) format.
 * @param scaling - The scaling factor to multiply the pixel values by.
 * @constructor - The Base64 image data URL.
 */
export async function tensorToImage(
  tensor: Tensor,
  scaling: number = 255,
): Promise<string> {
  const shape = getTensorShape(tensor);
  if (shape.length !== 3) {
    throw new Error("Invalid tensor shape. Expected 3D tensor.");
  }
  const [channels, width, height] = shape;
  if (channels !== 3 && channels !== 1) {
    throw new Error("Invalid number of channels. Expected 1 or 3.");
  }

  const arr = tensor as number[][][];

  // Flatten the 3D array into a 1D Uint8ClampedArray
  const imageData = new Uint8ClampedArray(width * height * 4); // RGBA format
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = (y * width + x) * 4;
      imageData[pixelIndex] = arr[0][y][x] * scaling;
      if (channels === 1) {
        // repeat the same value for all channels
        imageData[pixelIndex + 1] = imageData[pixelIndex];
        imageData[pixelIndex + 2] = imageData[pixelIndex];
      } else {
        imageData[pixelIndex + 1] = arr[1][y][x] * scaling;
        imageData[pixelIndex + 2] = arr[2][y][x] * scaling;
      }
      imageData[pixelIndex + 3] = 255; // Alpha channel (opaque)
    }
  }

  // Create a canvas
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");

  if (!ctx) {
    throw new Error("Failed to get 2D context");
  }

  // Create ImageData and put it on the canvas
  const imgData = new ImageData(imageData, width, height);
  ctx.putImageData(imgData, 0, 0);

  // Convert canvas to Base64 Data URL
  return canvas.toDataURL("image/png");
}
