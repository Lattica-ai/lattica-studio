import * as serverApi from './serverApi';
import * as workerApi from './workerApi';
import {
  Tensor,
  vecToTensor,
  tensor1DToVec,
  imageToTensor,
  tensorToImage,
  dumps_proto_tensor,
  loads_proto_tensor,
} from './tensorUtils';

import * as toolkit from './toolkitWrapper';

import { runInference } from './inference';
import { ClientModel, ClientBlock, ClientData } from './serialization/hom_op';

import { triggerEvkCopy } from './dev_utils';
import { initWasm, workerManagerSingleton, setLogCallback } from './wasm/wasm_worker_manager';
import { getDemoToken } from './serverApi';
export {
  initWasm,
  vecToTensor,
  tensor1DToVec,
  imageToTensor,
  tensorToImage,
  setLogCallback,
  getDemoToken,
}; // Re-export for frontend

export class LatticaQueryClient {
  public readonly jwt: string;
  public readonly beUrl: string;
  protected initialized: boolean = false;
  private serializedContext!: Uint8Array;
  private serializedContextRingSwitch!: Uint8Array;
  private secretKey!: [Uint8Array, Uint8Array];
  private secretKeyRingSwitch!: [Uint8Array, Uint8Array];
  private preprocessingBlock!: Uint8Array;
  private postprocessingBlock!: Uint8Array;
  private modelMetadata: Uint8Array | undefined;
  private ptAxisExternal: number | undefined;
  private asComplex?: boolean;

  constructor(jwt: string, beUrl?: string) {
    if (!jwt) throw new Error('JWT token is required');
    this.jwt = jwt;
    this.beUrl = beUrl ?? serverApi.BE_URL;
  }

  protected async getClientData(): Promise<Uint8Array> {
    const serializedClientData = await workerApi.get_user_init_data(this.jwt, this.beUrl);

    const clientData = ClientData.decode(serializedClientData);

    this.serializedContext = clientData.serializedContext;
    this.serializedContextRingSwitch = clientData.serialized_context_ring_switch;

    const model = ClientModel.decode(clientData.serializedModel);
    this.modelMetadata = model.preprocessingData;

    this.preprocessingBlock = ClientBlock.encode(model.preprocessBlock!).finish();
    this.ptAxisExternal = model.ptAxisExternal;

    this.postprocessingBlock = ClientBlock.encode(model.postprocessBlock!).finish();
    this.asComplex = model.asComplex;

    return clientData.serializedModel;
  }

  protected async generateKey(serializedHomseq: Uint8Array) {
    const [skRingSwitch, skRingSwitchCoefs, sk, skCoefs, evaluationKey] = await toolkit.generateKey(
      serializedHomseq,
      this.serializedContext,
      this.serializedContextRingSwitch,
    );

    this.secretKey = [sk, skCoefs];
    this.secretKeyRingSwitch = [skRingSwitch, skRingSwitchCoefs];


    console.log('PK size:', evaluationKey.length / (1024 * 1024), 'MB');

    return evaluationKey;
  }

  async init(): Promise<void> {
    await initWasm();

    workerManagerSingleton.logConsole('Getting context from server...');
    const serializedHomseq = await this.getClientData();
    workerManagerSingleton.logConsole('Context received.');

    const evaluationKey = await this.generateKey(serializedHomseq);

    workerManagerSingleton.logConsole('Uploading evaluation key...');
    await serverApi.uploadEk(this.jwt, this.beUrl, evaluationKey);
    workerManagerSingleton.logConsole('Evaluation key uploaded.');


    await workerApi.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }

  getModelMetadata(): Uint8Array | undefined {
    if (!this.initialized) throw new Error('Client must be initialized first');
    return this.modelMetadata;
  }

  async encrypt(pt: Tensor): Promise<Uint8Array> {
    if (!this.initialized) throw new Error('Client must be initialized first');

    workerManagerSingleton.logConsole(`Plain text shape: (${pt.shape})`);

    let serialized_pt = dumps_proto_tensor(pt);
    // apply first client block
    serialized_pt = await toolkit.applyClientBlock(
      this.preprocessingBlock,
      this.serializedContext,
      serialized_pt,
    );
    let secretKEyEnc: [Uint8Array, Uint8Array];
    let serializedContextEnc: Uint8Array;
    if (this.serializedContextRingSwitch) {
      serializedContextEnc = this.serializedContextRingSwitch;
      secretKEyEnc = this.secretKeyRingSwitch;
    }
    else {
      serializedContextEnc = this.serializedContext;
      secretKEyEnc = this.secretKey;
    }
    return await toolkit.encrypt(
      serializedContextEnc,
      secretKEyEnc,
      serialized_pt,
      true,
      this.ptAxisExternal,
    );
  }

  async apply_query(
    encryptedInput: Uint8Array,
    return_new_state: boolean = false,
  ): Promise<Uint8Array> {
    if (!this.initialized) throw new Error('Client must be initialized first');

    return await workerApi.apply_hom_pipeline(
      this.jwt,
      this.beUrl,
      encryptedInput,
      return_new_state,
    );
  }

  async decrypt(encryptedResult: Uint8Array): Promise<Tensor> {
    if (!this.initialized) throw new Error('Client must be initialized first');
    let decryptedOutput = await toolkit.decrypt(
      this.serializedContext,
      this.secretKey,
      encryptedResult,
      this.asComplex,
    );

    // apply last client block
    decryptedOutput = await toolkit.applyClientBlock(
      this.postprocessingBlock,
      this.serializedContext,
      decryptedOutput,
    );

    return loads_proto_tensor(decryptedOutput);
  }

  async runQuery(pt: Tensor): Promise<Tensor> {
    if (!this.initialized) throw new Error('Client must be initialized first');
    return await runInference(
      this.jwt,
      this.beUrl,
      this.serializedContext,
      this.secretKey,
      pt,
      this.preprocessingBlock,
      this.postprocessingBlock,
      this.ptAxisExternal,
      this.asComplex,
      this.serializedContextRingSwitch,
      this.secretKeyRingSwitch,
    );
  }
}

export class LatticaQueryClientLocalDev extends LatticaQueryClient {
  constructor(jwt: string) {
    super(jwt, 'http://localhost:3050');
  }

  async init(): Promise<void> {
    await initWasm();

    const serializedHomseq = await this.getClientData();

    const evaluationKey = await this.generateKey(serializedHomseq);

    const { s3Key } = await serverApi.sendHttpRequest(
      this.jwt,
      this.beUrl,
      '/api/token/get_pk_upload_url',
    );
    await serverApi.alertUploadComplete(this.jwt, this.beUrl, s3Key); // fake alert to change status of EVK

    // This is the only difference from the parent class
    await triggerEvkCopy(this.jwt, this.beUrl, evaluationKey, s3Key);

    await workerApi.preprocessEk(this.jwt, this.beUrl);

    this.initialized = true;
  }
}
