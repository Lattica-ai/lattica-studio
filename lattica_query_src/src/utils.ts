export type Tensor = number | Tensor[];

export const imageToMatrix = (image: HTMLImageElement): Promise<number[][][]> => {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
            reject("Unable to create 2D context");
            return;
        }

        ctx.drawImage(image, 0, 0, image.width, image.height);
        const { data, width, height } = ctx.getImageData(0, 0, image.width, image.height);

        const matrix = Array.from({ length: height }, (_, y) =>
            Array.from({ length: width }, (_, x) => {
                const index = (y * width + x) * 4;
                return [data[index], data[index + 1], data[index + 2], data[index + 3]];
            })
        );

        resolve(matrix);
    });
};

export const runInference = (context: Uint8Array|null, secretKey: [Uint8Array,Uint8Array]|null, encryptedInput: Uint8Array, homseq: Uint8Array|null): Uint8Array => {
    return new Uint8Array([11, 22, 33]);
};

export const to_serialized_lattica_tensor = (tensor: Tensor): Uint8Array =>{
    return new Uint8Array([1, 2, 3]);
}
