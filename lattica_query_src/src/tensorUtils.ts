import { DataType } from "./protos/generic_pb";

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

export const dataToTensor = (data, dtype) => {
  const buffer = data.slice().buffer; // Copy the data to a new buffer
  switch (dtype) {
    case DataType.DOUBLE:
      return new Float64Array(buffer);
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
};

export const reshapeWithStrides = (flatArray, shape, strides) => {
  // Create a function to access the elements in the flattened array using strides
  function getElement(indices) {
    let index = 0;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * strides[i];
    }
    return flatArray[index];
  }

  // Recursively create the multi-dimensional array based on shape and strides
  function createMultiDimensionalArray(dimIndex = 0, indices = []) {
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

// export const imageToMatrix = (
//   image: HTMLImageElement,
// ): Promise<number[][][]> => {
//   return new Promise((resolve, reject) => {
//     const canvas = document.createElement("canvas");
//     canvas.width = image.width;
//     canvas.height = image.height;
//     const ctx = canvas.getContext("2d");
//
//     if (!ctx) {
//       reject("Unable to create 2D context");
//       return;
//     }
//
//     ctx.drawImage(image, 0, 0, image.width, image.height);
//     const { data, width, height } = ctx.getImageData(
//       0,
//       0,
//       image.width,
//       image.height,
//     );
//
//     const matrix = Array.from({ length: height }, (_, y) =>
//       Array.from({ length: width }, (_, x) => {
//         const index = (y * width + x) * 4;
//         return [data[index], data[index + 1], data[index + 2], data[index + 3]];
//       }),
//     );
//
//     resolve(matrix);
//   });
// };