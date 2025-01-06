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

    let result = [];
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
 * @returns A promise resolving to a 2D array representing the RGBA matrix.
 */
export async function imageToTensor(
  base64: string,
  grayScale: boolean = false,
  scaling: number = 1,
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
      const rgbaMatrix = [];
      const { width, height, data } = imageData;
      const normalizedData = new Float64Array(data.length);
      for (let i = 0; i < data.length; i++) {
        normalizedData[i] = data[i] / scaling;
      }

      for (let y = 0; y < height; y++) {
        const row: number[][] = [];
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          if (grayScale) {
            row.push([normalizedData[idx]]);
          } else {
            row.push([
              normalizedData[idx],
              normalizedData[idx + 1],
              normalizedData[idx + 2],
            ]);
          }
        }
        rgbaMatrix.push(row);
      }

      resolve(rgbaMatrix);
    };

    img.onerror = (err) => reject(new Error(`Failed to load image: ${err}`));
  });
}
