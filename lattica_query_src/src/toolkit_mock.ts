export const generateKey = (serialized_homseq: Uint8Array, serialized_context: Uint8Array): {secretKey: [Uint8Array,Uint8Array], evaluationKey: Uint8Array} => {
return {
    secretKey: [new Uint8Array([3,2,1]), new Uint8Array([31,21,11])],
    evaluationKey: new Uint8Array([6,5,4])
};
}

export const encrypt = (serialized_context: Uint8Array|null,
        serialized_sk: [Uint8Array, Uint8Array]|null,
        serialized_pt: Uint8Array,
        pack_for_transmission: boolean = false): Uint8Array => {
return new Uint8Array([10,20,30]);
}

export const decrypt = (context: Uint8Array|null, secretKey: [Uint8Array,Uint8Array]|null, encryptedResult: Uint8Array): Uint8Array => {
return new Uint8Array([15,25,35]);
}