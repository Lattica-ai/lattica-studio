import {initWasm} from "../index";

import * as toolkit from "../toolkitWrapper";

async function _loadDictFromFile(url: any) {
    const response = await fetch(url);
    const dict = await response.json();

    function decodeValue(value: any) {
        if (value && typeof value === 'object') {
            if (value.__bytes__) {
                return Uint8Array.from(atob(value.__bytes__), c => c.charCodeAt(0));
            }
            if (value.__tuple__) {
                return value.__tuple__.map(decodeValue);
            }
        }
        return value;
    }

    // Decode any encoded values
    Object.keys(dict).forEach(key => {
        console.log("Decoding key:", key);
        dict[key] = decodeValue(dict[key]);
    });
    return dict;
}

async function _demo_apply_client_block() {
    console.log("Loading dictionary from file...");
    const dict = await _loadDictFromFile('./src/examples/data/apply_client_block.json');
    console.log("Dictionary loaded:", dict);
    const result: any  = await toolkit.applyClientBlock(
        dict['serialized_block'],
        dict['serialized_context'],
        dict['serialized_pt']
    );
    const expected = dict['res'];
    console.log("Result:", result);
    console.log("Expected:", expected);
    if (JSON.stringify(result) !== JSON.stringify(expected))
        throw new Error("Result does not match expected");
}

async function _demo_enc() {
    console.log("Loading dictionary from file...");
    const dict = await _loadDictFromFile('./src/examples/data/enc.json');
    console.log("Dictionary loaded:", dict);
    const result: any  = await toolkit.encrypt(
        dict['serialized_context'],
        [dict['serialized_sk'], dict['serialized_sk']],
        dict['serialized_pt'],
        dict['pack_for_transmission']
    );
    const expected = dict['res'];
    console.log("Result:", result);
    console.log("Expected:", expected);
    // NOTE: The result is not deterministic, so we only check the length
    if (result.length !== expected.length)
        throw new Error("Result does not match expected");
}

async function _demo_dec() {
    console.log("Loading dictionary from file...");
    const dict = await _loadDictFromFile('./src/examples/data/dec.json');
    console.log("Dictionary loaded:", dict);
    const result: any  = await toolkit.decrypt(
        dict['serialized_context'],
        dict['serialized_sk'],
        dict['serialized_ct']
    );
    const expected = dict['res'];
    console.log("Result:", result);
    console.log("Expected:", expected);
    if (JSON.stringify(result) !== JSON.stringify(expected))
        throw new Error("Result does not match expected");
}

async function _demo_generate_key() {
    console.log("Loading dictionary from file...");
    const dict = await _loadDictFromFile('./src/examples/data/generate_key.json');
    console.log("Dictionary loaded:", dict);
    const result: any  = await toolkit.generateKey(
        dict['serialized_homseq'],
        dict['serialized_context']
    );
    const expected = [
        dict['res'][0][0],
        dict['res'][0][1],
        dict['res'][1]
    ]
    console.log("Result:", result);
    console.log("Expected:", expected);
    if (JSON.stringify(result) !== JSON.stringify(expected))
        throw new Error("Result does not match expected");
}



(async () => {
      await initWasm();
      await _demo_apply_client_block();
      await _demo_enc();
      await _demo_dec();
      // await _demo_generate_key();
})();
