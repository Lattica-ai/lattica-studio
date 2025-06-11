declare module "ndarray" {
  export interface NdArray<T = any> {
    data: T;
    shape: number[];
    stride: number[];
    offset: number;
    get(...args: number[]): number;
    set(...args: number[]): void;

    /** Derived string from underlying TypedArray, e.g., "float64", "int32" */
    dtype: string;
  }

  export default function ndarray<T = any>(
    data: T,
    shape?: number[],
    stride?: number[],
    offset?: number,
  ): NdArray<T>;
}
