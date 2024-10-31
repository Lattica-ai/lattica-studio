export class LatticaQueryClient {
    private readonly jwt: string;
    private image: HTMLImageElement | null = null;
    private initialized: boolean = false;
    private queryApplied: boolean = false;

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

    async encrypt(image: HTMLImageElement): Promise<void> {
        if (!this.initialized) throw new Error('Client must be initialized first');
        if (!image) throw new Error('Image is required');

        // const matrix = convertImageToMatrix(image)

        // this.encryptedInput = toolkit.encrypt(matrix)

        this.image = image; // todo: remove after WASM integration
        await this.delay();
    }

    async apply_query(): Promise<void> {
        if (!this.image) throw new Error('Image must be encrypted first');

        // this.encryptedOutput = homUtils.runInference(this.context, this.secretKey, this.encryptedInput, this.homseq)

        await this.delay();
        this.queryApplied = true;
    }

    async decrypt(): Promise<HTMLImageElement> {
        if (!this.image) throw new Error('Query must be applied first');
        if (!this.queryApplied) throw new Error('Query must be applied before decryption');

        // const decriptedOutput = toolkit.decrypt(this.secretKey, this.encryptedOutput)

        // return convertMatrixToImage(decriptedOutput)

        await this.delay();
        return this.image;
    }

    // todo: remove after WASM toolkit is integrated!
    private delay(ms: number = 300): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}