import { LatticaQueryClient } from '../index';

// Mock HTMLImageElement since we're in Node environment
class MockHTMLImageElement {
    src: string = '';
    constructor() {
        // Mock implementation
    }
}

describe('LatticaQueryClient', () => {
    let client: LatticaQueryClient;
    let mockImage: HTMLImageElement;
    const mockJwt = 'mock-jwt-token';

    beforeEach(async () => {
        // Setup before each test
        client = new LatticaQueryClient(mockJwt);
        await client.init();
        mockImage = new MockHTMLImageElement() as unknown as HTMLImageElement;
    });

    describe('constructor', () => {
        it('should create instance with jwt', () => {
            expect(client).toBeInstanceOf(LatticaQueryClient);
        });

        it('should throw error if jwt is empty', () => {
            expect(() => new LatticaQueryClient('')).toThrow('JWT token is required');
        });
    });

    describe('init', () => {
        it('should set initialized to true', async () => {
            const newClient = new LatticaQueryClient(mockJwt);
            await newClient.init();
            expect((newClient as any).initialized).toBe(true);
        });
    });

    describe('encrypt', () => {
        it('should encrypt image successfully', async () => {
            const result = await client.encrypt(mockImage);
            expect(result).toBe(true);
        });

        it('should store the image internally', async () => {
            await client.encrypt(mockImage);
            expect(client.image).toBe(mockImage);
        });

        it('should throw error if image is null', async () => {
            await expect(client.encrypt(null as unknown as HTMLImageElement))
                .rejects
                .toThrow('Image is required');
        });

        it('should throw error if init was not called', async () => {
            const newClient = new LatticaQueryClient(mockJwt);
            await expect(newClient.encrypt(mockImage))
                .rejects
                .toThrow('Client must be initialized first');
        });
    });

    describe('apply_query', () => {
        it('should apply query successfully', async () => {
            await client.encrypt(mockImage);
            const result = await client.apply_query();
            expect(result).toBe(true);
        });

        it('should throw error if image is not encrypted first', async () => {
            await expect(client.apply_query())
                .rejects
                .toThrow('Image must be encrypted first');
        });
    });

    describe('decrypt', () => {
        it('should decrypt and return image', async () => {
            await client.encrypt(mockImage);
            await client.apply_query();
            const result = await client.decrypt();
            expect(result).toBe(mockImage);
        });

        it('should throw error if query not applied', async () => {
            await client.encrypt(mockImage);
            await expect(client.decrypt())
                .rejects
                .toThrow('Query must be applied before decryption');
        });
    });
});