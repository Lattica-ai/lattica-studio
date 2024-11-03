export const getUserInitData = async (): Promise<{ serializedContext: Uint8Array, serializedHomseq: Uint8Array }> => {
    return {
        serializedContext: new Uint8Array([1, 2, 3]),
        serializedHomseq: new Uint8Array([4, 5, 6])
    };
}

export const uploadEk = async (jwt: string, evaluationKey: Uint8Array): Promise<void> => {
    console.log('Uploading evaluation key');
}

export const preprocessEk = async (): Promise<void> => {
    console.log('Preprocessing evaluation key');
}