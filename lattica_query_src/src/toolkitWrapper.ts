import { workerManagerSingleton } from './wasm/wasm_worker_manager';

export const applyClientBlock = async (
  serialized_block: Uint8Array,
  serialized_context: Uint8Array,
  serialized_pt: Uint8Array,
) => {
  workerManagerSingleton.logConsole('Applying client block...');
  const res = await workerManagerSingleton.callWorker('apply_client_block', {
    block: serialized_block,
    context: serialized_context,
    pt: serialized_pt,
  });
  workerManagerSingleton.logConsole('Client block applied.');
  return res;
};

export const encrypt = async (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_pt: Uint8Array,
  pack_for_transmission: boolean = false,
  pt_axis_external?: number,
  custom_state_name?: string,
) => {
  workerManagerSingleton.logConsole('Encrypting plain text...');
  const res = await workerManagerSingleton.callWorker('enc', {
    context: serialized_context,
    sk: serialized_sk,
    pt: serialized_pt,
    pack_for_transmission,
    pt_axis_external: pt_axis_external ?? -1, // wasm expects -1 for undefined
    custom_state_name: custom_state_name ?? '', // wasm expects '' for undefined
  });
  workerManagerSingleton.logConsole('Encrypted.');
  return res;
};

export const decrypt = async (
  serialized_context: Uint8Array,
  serialized_sk: [Uint8Array, Uint8Array],
  serialized_ct: Uint8Array,
  as_complex: boolean = false,
) => {
  workerManagerSingleton.logConsole('Decrypting ciphertext...');
  const res = await workerManagerSingleton.callWorker('dec', {
    context: serialized_context,
    sk: serialized_sk[1],
    ct: serialized_ct,
    as_complex,
  });
  workerManagerSingleton.logConsole('Decrypted.');
  return res;
};

export const generateKey = async (
  serialized_homseq: Uint8Array,
  serialized_context: Uint8Array,
  serialized_context_ring_switch: Uint8Array,
) => {
  workerManagerSingleton.logConsole('Generating keys...');
  const res = await workerManagerSingleton.callWorker('generate_key', {
    homseq: serialized_homseq,
    context: serialized_context,
    context_ring_switch: serialized_context_ring_switch,
  });
  workerManagerSingleton.logConsole('Keys generated.');
  return res;
};
