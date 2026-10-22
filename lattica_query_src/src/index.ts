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
import { ClientModel, ClientBlock, ClientData, ExtendedContext, ExtendedSecretKey } from '../lattica_query/serialization/generated/hom_op';
import { TensorHolder } from '../lattica_query/serialization/generated/generic';

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
  private extendedContext!: Uint8Array;
  private context!: Uint8Array;
  private ringSwitchContext?: Uint8Array;
  private secretKey!: [Uint8Array, Uint8Array];
  private ringSwitchSecretKey?: [Uint8Array, Uint8Array];
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

    this.extendedContext = clientData.serializedExtendedContext;
    
    const extendedContext = ExtendedContext.decode(clientData.serializedExtendedContext);
    this.context = extendedContext.serializedContext;
    this.ringSwitchContext = extendedContext.serializedRingSwitchContext;

    const model = ClientModel.decode(clientData.serializedModel);
    this.modelMetadata = model.preprocessingData;

    this.preprocessingBlock = ClientBlock.encode(model.preprocessBlock!).finish();
    this.ptAxisExternal = model.ptAxisExternal;

    this.postprocessingBlock = ClientBlock.encode(model.postprocessBlock!).finish();
    this.asComplex = model.asComplex;

    return clientData.serializedModel;
  }

  protected async generateKey(serializedHomseq: Uint8Array) {
    const [serializedExtendedSecretKey, evaluationKey] = await toolkit.generateKey(
      serializedHomseq,
      this.extendedContext,
    );

    const extendedSecretKey = ExtendedSecretKey.decode(serializedExtendedSecretKey);
    
    // Extract main secret key
    const sk = TensorHolder.encode(extendedSecretKey.secretKey!.sk!).finish();
    const skCoefs = TensorHolder.encode(extendedSecretKey.secretKey!.skCoefs!).finish();
    this.secretKey = [sk, skCoefs];
    
    // Extract ring switch secret key if it exists
    if (extendedSecretKey.ringSwitchSecretKey) {
      const ringSwitchSk = TensorHolder.encode(extendedSecretKey.ringSwitchSecretKey.sk!).finish();
      const ringSwitchSkCoefs = TensorHolder.encode(extendedSecretKey.ringSwitchSecretKey.skCoefs!).finish();
      this.ringSwitchSecretKey = [ringSwitchSk, ringSwitchSkCoefs];
    }

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
    const encContext = this.ringSwitchContext ?? this.context;
    const encSecretKey = this.ringSwitchSecretKey ?? this.secretKey;
    serialized_pt = await toolkit.applyClientBlock(
      this.preprocessingBlock,
      encContext,
      serialized_pt,
    );

    return await toolkit.encrypt(
      encContext,
      encSecretKey,
      serialized_pt,
      true,
      this.ptAxisExternal,
    );
  }

  async apply_query(encryptedInput: Uint8Array): Promise<Uint8Array> {
    if (!this.initialized) throw new Error('Client must be initialized first');

    return await workerApi.apply_hom_pipeline(
      this.jwt,
      this.beUrl,
      encryptedInput,
    );
  }

  async decrypt(encryptedResult: Uint8Array): Promise<Tensor> {
    if (!this.initialized) throw new Error('Client must be initialized first');
    
    let decryptedOutput = await toolkit.decrypt(
      this.context,
      this.secretKey,
      encryptedResult,
      this.asComplex,
    );

    // apply last client block
    decryptedOutput = await toolkit.applyClientBlock(
      this.postprocessingBlock,
      this.context,
      decryptedOutput,
    );

    return loads_proto_tensor(decryptedOutput);
  }

  async runQuery(pt: Tensor): Promise<Tensor> {
    if (!this.initialized) throw new Error('Client must be initialized first');
    return await runInference(
      this.jwt,
      this.beUrl,
      this.context,
      this.secretKey,
      pt,
      this.preprocessingBlock,
      this.postprocessingBlock,
      this.ptAxisExternal,
      this.asComplex,
      this.ringSwitchContext,
      this.ringSwitchSecretKey,
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
