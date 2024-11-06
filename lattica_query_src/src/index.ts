import * as toolkit from "./toolkit_mock";
import * as server from "./server";
import * as worker_api from "./worker_api";
import * as utils from "./utils";
import { Tensor } from "./utils";
import * as fs from "fs";
import * as dotenv from "dotenv";
dotenv.config();

export class LatticaQueryClient {
  private readonly jwt: string;
  private image: HTMLImageElement | null = null;
  private initialized: boolean = false;
  private serializedContext: Buffer | null = null;
  private serializedHomseq: Buffer | null = null;
  private secretKey: [Buffer, Buffer] | null = null;

  constructor(jwt: string) {
    if (!jwt) throw new Error("JWT token is required");
    this.jwt = jwt;
  }

  async init(): Promise<void> {
    const initData = await worker_api.get_user_init_data(this.jwt);
    this.serializedContext = initData.serializedContext;
    this.serializedHomseq = initData.serializedHomseq;

    const { secretKey, evaluationKey } = toolkit.generateKey(
      this.serializedHomseq,
      this.serializedContext,
    );
    this.secretKey = secretKey;

    const ekFilePath = "my_ek.lek";
    fs.writeFileSync(ekFilePath, evaluationKey);
    await server.uploadEk(this.jwt, ekFilePath);
    fs.unlinkSync(ekFilePath);

    await worker_api.preprocessEk(this.jwt);

    this.initialized = true;
    await this.delay();
  }

  async encrypt(pt: utils.Tensor): Promise<Buffer> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    // const pt = await utils.imageToMatrix(image)

    const serialized_pt = toolkit.dumps_proto_tensor(pt);
    const encryptedInput = toolkit.encrypt(
      this.serializedContext,
      this.secretKey,
      serialized_pt,
      true,
    );

    await this.delay();
    return encryptedInput;
  }

  async apply_query(encryptedInput: Buffer): Promise<Uint8Array> {
    if (!this.initialized) throw new Error("Client must be initialized first");

    const encryptedOutput = await utils.runInference(
      this.jwt,
      this.serializedContext,
      this.secretKey,
      encryptedInput,
      this.serializedHomseq,
    );

    await this.delay();
    return encryptedOutput;
  }

  async decrypt(encryptedResult: Buffer): Promise<HTMLImageElement | null> {
    if (!this.initialized) throw new Error("Client must be initialized first");
    const decryptedOutput = toolkit.decrypt(
      this.serializedContext,
      this.secretKey,
      encryptedResult,
    );

    await this.delay();
    return this.image;
  }

  // todo: remove after WASM toolkit is integrated!
  private delay(ms: number = 300): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

async function processImage(pt: Tensor) {
  await client.init();
  console.log("Client initialized:", client);
  const ct = await client.encrypt(pt);
  console.log("Encrypted:", ct);
  const ct_res = await client.apply_query(ct);
  console.log("Query result:", ct_res);
  const pt_res = await client.decrypt(ct_res);
  console.log("Decrypted:", pt_res);
  return pt_res;
}

const client = new LatticaQueryClient(
  "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IkZTRzBSMmVuWFYzZXJSR0pMaFFidSJ9.eyJ0b2tlbklkIjoiMGU2NGEwNmEtYmZkNC00MTY2LWJiZDktOTkzYmI1MWMyZGNjIiwiaXNzIjoiaHR0cHM6Ly9kZXYtZjVrZ3Iwa25yd2pvYWF3MS51cy5hdXRoMC5jb20vIiwic3ViIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3ZAY2xpZW50cyIsImF1ZCI6Imh0dHBzOi8vbXlhcGkvYXBpIiwiaWF0IjoxNzMwODM5NTY1LCJleHAiOjE3MzM0MzE1NjUsInNjb3BlIjoidXNlciBkZWZhdWx0IiwiZ3R5IjoiY2xpZW50LWNyZWRlbnRpYWxzIiwiYXpwIjoiUFFJUVVpN24zbUpqSnBNdmFKRkQzQ09kTzRwMzVCc3YiLCJwZXJtaXNzaW9ucyI6WyJ1c2VyIiwiZGVmYXVsdCJdfQ.6grZW1Qek8lG8ciDEQ7d0MjmStuNYbVymzWp1Kbcl8TIxL8HlQE6FJNFKx0mOQlzwZuirzm-achQoHp2Uhn9lcBJ00vWWDmC0j7PqugYFo2PegVBr58xp8cuCxueAlgbZbCOZoqmZ61PHHVEJsd_THWOu6GNTLdTxy7SbKB8W756uofVK-hyFlP3DbP2-bWuzWI6QuwijnfFbOsRGd-U_zYJ_TVcasjLetbEpPWiHaxxOoBdRTDdPuvgINbfiGyBZcZySG5cGsqKAJpvcSR_KPlya1dJwz6qBn1-P061EKn3VWWG7tEm3JJaKnOnQI6VOE1G0cYwcHqKsexmgjM7EA",
);
console.log("Client created:", client);

const pt = [
  [
    [1, 2],
    [3, 4],
  ],
  [
    [5, 6],
    [7, 8],
  ],
];

processImage(pt).then((pt_res) => {
  console.log("Result:", pt_res);
});
