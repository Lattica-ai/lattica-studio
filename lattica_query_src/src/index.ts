export class LatticaQueryClient {
    private readonly jwt: string;
    private image: string | null = null;
    private initialized: boolean = false;

    constructor(jwt: string) {
        if (!jwt) throw new Error('JWT token is required');
        this.jwt = jwt;
    }

    async init(): Promise<void> {
        // const initData = await server.getUserInitData();
        // this.context = initData.serializedContext;
        // this.homseq = initData.serializedHomseq;

        // const {secretKey, evaluationKey} = toolkit.generateKey(this.serialized_homseq, this.serialized_context)
        // this.secretKey = secretKey;

        // await server.uploadEk(this.jwt, evaluationKey)

        // wait server.preprocessEk()

        this.initialized = true;
        await this.delay();
    }

    async encrypt(encodedImage: string): Promise<Uint8Array> {
        if (!this.initialized) throw new Error('Client must be initialized first');
        if (!encodedImage) throw new Error('Image is required');

        // const matrix = convertImageToMatrix(image)

        // this.encryptedInput = toolkit.encrypt(matrix)

        this.image = encodedImage; // todo: remove after WASM integration
        await this.delay();
        return new Uint8Array([1, 2, 3]);
    }

    async apply_query(encryptedInput: Uint8Array): Promise<Uint8Array> {
        if (!this.initialized) throw new Error('Client must be initialized first');

        // this.encryptedOutput = homUtils.runInference(this.context, this.secretKey, this.encryptedInput, this.homseq)

        await this.delay();
        return new Uint8Array([4, 5, 6]);
    }

    async decrypt(encryptedResult: Uint8Array): Promise<string> {
        if (!this.initialized) throw new Error('Client must be initialized first');

        // const decriptedOutput = toolkit.decrypt(this.secretKey, this.encryptedOutput)

        // return convertMatrixToImage(decriptedOutput)

        await this.delay();
        return this.image as string;
    }

    // todo: remove after WASM toolkit is integrated!
    private delay(): Promise<void> {
        const ms = Math.floor(Math.random() * (5000 - 2000 + 1)) + 2000;
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}