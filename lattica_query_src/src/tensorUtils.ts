import { DataType, TensorHolder } from "../lattica_query/serialization/generated/generic";
import ndarray from "ndarray";
export type Tensor = import("ndarray").NdArray;

function dtypeToTypedArray(dtype: DataType, buffer: ArrayBuffer) {
  switch (dtype) {
    case DataType.FLOAT:
      return new Float32Array(buffer);
    case DataType.DOUBLE:
      return new Float64Array(buffer);
    case DataType.INT32:
      return new Int32Array(buffer);
    case DataType.INT64:
      return new BigInt64Array(buffer);
    case DataType.BOOL:
      return new Uint8Array(buffer);
    default:
      throw new Error(`Unsupported dtype ${dtype}`);
  }
}

function dtypeStringToEnum(dtype: string): DataType {
  switch (dtype) {
    case "float32":
      return DataType.FLOAT;
    case "float64":
      return DataType.DOUBLE;
    case "int32":
      return DataType.INT32;
    case "bigint64":
      return DataType.INT64;
    case "uint8":
      return DataType.BOOL;
    default:
      return DataType.UNRECOGNIZED;
  }
}

export function loads_proto_tensor(serializedTensorHolder: Uint8Array): Tensor {
  const tensorHolder = TensorHolder.decode(serializedTensorHolder);
  const buffer = tensorHolder.data.slice().buffer;
  const typedArray = dtypeToTypedArray(tensorHolder.dtype, buffer);
  return ndarray(typedArray, tensorHolder.sizes, tensorHolder.strides);
}

export function dumps_proto_tensor(arr: Tensor): Uint8Array {
  const data = new Uint8Array(
    arr.data.buffer,
    arr.data.byteOffset || 0,
    arr.data.byteLength || arr.data.length * arr.data.BYTES_PER_ELEMENT,
  );

  const proto = TensorHolder.create({
    sizes: arr.shape,
    strides: arr.stride,
    data,
    dtype: dtypeStringToEnum(arr.dtype),
  });

  return TensorHolder.encode(proto).finish();
}

export function vecToTensor(vec: number[]): Tensor {
  return ndarray(new Float64Array(vec), [vec.length]);
}

export function tensor1DToVec(tensor: Tensor): number[] {
  if (tensor.shape.length !== 1) {
    throw new Error("Invalid tensor shape. Expected 1D tensor.");
  }
  const vec: number[] = [];
  for (let i = 0; i < tensor.shape[0]; i++) {
    vec.push(Number(tensor.get(i)));
  }
  return vec;
}

export async function imageToTensor(
  base64: string,
  grayScale: boolean = false,
  normalize: boolean = true,
  targetWidth?: number,
  targetHeight?: number,
): Promise<Tensor> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = base64;

    img.onload = () => {
      const width = targetWidth ? targetWidth : img.width;
      const height = targetHeight ? targetHeight : img.height;

      console.log(
        "Resize image:",
        img.width,
        img.height,
        " => ",
        width,
        height,
      );

      // Draw image onto a canvas
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0, width, height);

      // Get pixel data (RGBA)
      const imageData = ctx.getImageData(0, 0, width, height);
      const rgba = imageData.data; // Uint8ClampedArray

      // Convert to [C, H, W]
      const C = grayScale ? 1 : 3;
      const H = height;
      const W = width;
      const data = new Float64Array(C * H * W);

      const factor = normalize ? 255 : 1;

      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          const i = (h * W + w) * 4; // RGBA stride
          const pixelIndex = h * W + w;
          data[0 * H * W + pixelIndex] = rgba[i] / factor;

          if (!grayScale) {
            data[1 * H * W + pixelIndex] = rgba[i + 1] / factor;

            data[2 * H * W + pixelIndex] = rgba[i + 2] / factor;
          }
        }
      }

      resolve(ndarray(data, [C, H, W]));
    };
    img.onerror = (err) => reject(new Error(`Failed to load image: ${err}`));
  });
}

export async function tensorToImage(
  tensor: Tensor,
  normalized: boolean = true,
): Promise<string> {
  if (tensor.shape.length !== 3) {
    throw new Error("Invalid tensor shape. Expected 3D tensor.");
  }
  const [C, H, W] = tensor.shape;
  if (C !== 3 && C !== 1) {
    throw new Error("Invalid number of channels. Expected 1 or 3.");
  }

  // Prepare RGBA pixel array
  const rgba = new Uint8ClampedArray(H * W * 4);
  const factor = normalized ? 255 : 1;

  for (let h = 0; h < H; h++) {
    for (let w = 0; w < W; w++) {
      const idx = (h * W + w) * 4;
      rgba[idx] = Number(tensor.get(0, h, w)) * factor;
      if (C === 1) {
        rgba[idx + 1] = rgba[idx];
        rgba[idx + 2] = rgba[idx];
      } else {
        rgba[idx + 1] = Number(tensor.get(1, h, w)) * factor;
        rgba[idx + 2] = Number(tensor.get(2, h, w)) * factor;
      }
      rgba[idx + 3] = 255; // Alpha channel (opaque)
    }
  }

  // Put pixels onto canvas
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  const imageData = new ImageData(rgba, W, H);
  ctx.putImageData(imageData, 0, 0);

  return canvas.toDataURL("image/png");
}
