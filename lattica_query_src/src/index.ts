import * as toolkit from "./toolkit_mock";
import * as server from "./server";
import * as utils from "./utils";
import {Tensor} from "./utils";

export class LatticaQueryClient {
    private readonly jwt: string;
    private image: HTMLImageElement | null = null;
    private initialized: boolean = false;
    private serialized_context: Uint8Array | null = null;
    private serialized_homseq: Uint8Array | null = null;
    private secretKey: [Uint8Array,Uint8Array] | null = null;

    constructor(jwt: string) {
        if (!jwt) throw new Error('JWT token is required');
        this.jwt = jwt;
    }

    async init(): Promise<void> {
        const initData = await server.getUserInitData();
        this.serialized_context = initData.serializedContext;
        this.serialized_homseq = initData.serializedHomseq;

        const {secretKey, evaluationKey} = toolkit.generateKey(this.serialized_homseq, this.serialized_context)
        this.secretKey = secretKey;

        await server.uploadEk(this.jwt, evaluationKey)

        await server.preprocessEk()

        this.initialized = true;
        await this.delay();
    }

    async encrypt(pt: utils.Tensor): Promise<Uint8Array> {
        if (!this.initialized) throw new Error('Client must be initialized first');

        // const pt = await utils.imageToMatrix(image)

        const serialized_pt = utils.to_serialized_lattica_tensor(pt)
        const encryptedInput = toolkit.encrypt(this.serialized_context, this.secretKey, serialized_pt, true)

        await this.delay();
        return encryptedInput;
    }

    async apply_query(encryptedInput: Uint8Array): Promise<Uint8Array> {
        if (!this.initialized) throw new Error('Client must be initialized first');

        const encryptedOutput = utils.runInference(this.serialized_context, this.secretKey, encryptedInput, this.serialized_homseq)

        await this.delay();
        return encryptedOutput;
    }

    async decrypt(encryptedResult: Uint8Array): Promise<HTMLImageElement | null> {
        if (!this.initialized) throw new Error('Client must be initialized first');
        const decryptedOutput = toolkit.decrypt(this.serialized_context, this.secretKey, encryptedResult)

        await this.delay();
        return this.image;
    }

    // todo: remove after WASM toolkit is integrated!
    private delay(ms: number = 300): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

async function processImage(pt: Tensor) {
    await client.init();
    console.log('Client initialized:', client);
    const ct = await client.encrypt(pt);
    console.log('Encrypted:', ct);
    const ct_res = await client.apply_query(ct);
    console.log('Query result:', ct_res);
    const pt_res = await client.decrypt(ct_res);
    console.log('Decrypted:', pt_res);
    return pt_res;
}

const client = new LatticaQueryClient('your-jwt-token');
console.log('Client created:', client);

const pt = [
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
];

processImage(pt).then(pt_res => {
    console.log('Result:', pt_res);
});