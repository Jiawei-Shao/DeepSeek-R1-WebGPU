import {
  AutoTokenizer,
  env,
} from "@huggingface/transformers";
import ort from 'onnxruntime-web/webgpu'

class LLM {
  model_id = "DeepSeek-R1-Distill-Qwen-1.5B-ONNX";

  model_data_file_name = `model_q4f16.onnx`;
  model_data_file = `/models/${this.model_id}/onnx/model_q4f16.onnx`;
  model_config_file_name = `config.json`;
  model_config_file = `/models/${this.model_id}/config.json`;

  tokenizer = undefined;
  inferenceSession = undefined;

  kv_dims = [];
  num_hidden_layers = 0;

  eos = 0n;
  end_thinking_token_id = 0;

  async init() {
    ort.env.wasm.wasmPaths = '/dist/';
    ort.env.wasm.numThreads = 1;

    const callback = (x) => {
      self.postMessage(x);
    };

    this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id, callback);

    // 151648: <think>
    // 151649: </think>
    const [END_THINKING_TOKEN_ID] = llm.tokenizer.encode(
      "</think>",
      { add_special_tokens: false },
    );
    this.end_thinking_token_id = END_THINKING_TOKEN_ID;

    const modelBytes = await this.fetchAndCache(this.model_data_file, callbackInfo => {
      callback({
        name: this.model_id,
        file: this.model_data_file_name,
        ...callbackInfo,
      })
    });
    let modelSize = modelBytes.byteLength;
    console.log(`${Math.round(modelSize / 1024 / 1024)} MB`);
    const inferenceSessionOptions = {
      executionProviders: ["webgpu"],
      preferredOutputLocation: {},
    };
    const jsonBytes = await this.fetchAndCache(this.model_config_file, callbackInfo => {
      callback({
        name: this.model_id,
        file: this.model_config_file_name,
        ...callbackInfo,
      })
    });
    const textDecoder = new TextDecoder();
    const modelConfig = JSON.parse(textDecoder.decode(jsonBytes));
    for (let i = 0; i < modelConfig.num_hidden_layers; ++i) {
      inferenceSessionOptions.preferredOutputLocation
          [`present.${i}.key`] = 'gpu-buffer';
      inferenceSessionOptions.preferredOutputLocation
          [`present.${i}.value`] = 'gpu-buffer';
    }
    this.inferenceSession =
      await ort.InferenceSession.create(modelBytes, inferenceSessionOptions);

    this.eos = BigInt(modelConfig.eos_token_id);
    this.num_hidden_layers = modelConfig.num_hidden_layers;
    this.kv_dims =
        [1, modelConfig.num_key_value_heads, 0,
         modelConfig.hidden_size / modelConfig.num_attention_heads];
  }

  argmax(t) {
    const arr = t.data;
    const start = t.dims[2] * (t.dims[1] - 1);
    let max = arr[start];
    let maxidx = 0;

    for (let i = 0; i < t.dims[2]; i++) {
        const val = arr[i + start];
        if (!isFinite(val)) {
            throw new Error("found infinitive in logits");
        }
        if (val > max) {
            max = arr[i + start];
            maxidx = i;
        }
    }
    return maxidx;
}

async TokenizePrompt(prompt) {
  const promptTokenizerResult = await this.tokenizer(
      prompt, { return_tensor: false, padding: true, truncation: true });
  const promptTokens = promptTokenizerResult.input_ids;
  return new ort.Tensor(
      'int64', BigInt64Array.from(promptTokens.map(BigInt)),
      [1, promptTokens.length]);
}

TokensToText(tokens) {
  return this.tokenizer.decode(
      tokens, { skip_special_tokens: false, });
}

update_kv_cache(outputs, feed) {
    for (const name in outputs) {
        if (name.startsWith('present')) {
            let newName = name.replace('present', 'past_key_values');
            const t = feed[newName];
            if (t.location === 'gpu-buffer') {
                t.dispose();
            }
            feed[newName] = outputs[name];
        }
    }
  }

  async fetchAndCache(url, callback) {
    callback({
      status: 'initiate',
    });

    try {
        const cache = await caches.open("onnx");
        let response = await cache.match(url);
        const cacheMiss = response === undefined;
        if (cacheMiss) {
          callback({
            status: 'download',
          });
          response = await fetch(url);
        }

        const data = await this.readResponse(response, callback);

        if (cacheMiss) {
          try {
              await cache.put(url, new Response(data, {
                headers: networkResponse.headers
              }));
          } catch (error) {
              console.error(error);
          }
        }

        callback({
          status: 'done',
        });

        return data;
    } catch (error) {
        console.log(`can't fetch ${url}`);
        throw error;
    }
  }

  // Referenced from transformers.js/src/utils/hub.js
  async readResponse(response, callback) {
    const contentLength = response.headers.get('Content-Length');
    let total = parseInt(contentLength ?? '0');
    let buffer = new Uint8Array(total);
    let loaded = 0;

    const reader = response.body.getReader();
    async function read() {
        const { done, value } = await reader.read();
        if (done) return;

        let newLoaded = loaded + value.length;
        if (newLoaded > total) {
            total = newLoaded;

            // Adding the new data will overflow buffer.
            // In this case, we extend the buffer
            let newBuffer = new Uint8Array(total);

            // copy contents
            newBuffer.set(buffer);

            buffer = newBuffer;
        }
        buffer.set(value, loaded)
        loaded = newLoaded;

        const progress = (loaded / total) * 100;

        // Call your function here
        callback({
            status: 'progress',
            progress: progress,
            loaded: loaded,
            total: total,
        })

        return read();
    }

    // Actually read
    await read();

    return buffer;
  }

  async query(messages, max_output_tokens, callback_function = null, token_callback_function = null) {
    const inferenceInputIds = this.tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
    });

    let feed = {};
    const empty = new Uint16Array();
    for (let i = 0; i < this.num_hidden_layers; ++i) {
        feed[`past_key_values.${i}.key`] = new ort.Tensor('float16', empty, this.kv_dims);
        feed[`past_key_values.${i}.value`] = new ort.Tensor('float16', empty, this.kv_dims);
    }

    feed['input_ids'] = inferenceInputIds;
    const output_tokens = [];
    output_tokens.push(...inferenceInputIds.data);

    let seqlen = output_tokens.length;
    const input_len = inferenceInputIds.size;
    feed['position_ids'] = new ort.Tensor(
        'int64', BigInt64Array.from({ length: input_len },
            (_, i) => BigInt(seqlen - input_len + i)),
            [1, input_len]);

    let last_token = 0n;
    while (last_token != this.eos && seqlen < max_output_tokens) {
        
        seqlen = output_tokens.length;
        feed['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen }, () => 1n), [1, seqlen]);

        const outputs = await this.inferenceSession.run(feed);
        last_token = BigInt(this.argmax(outputs.logits));

        if (last_token === this.eos) {
          break;
        }

        output_tokens.push(last_token);

        const text = this.TokensToText([last_token]);
        if (token_callback_function && callback_function) {
          token_callback_function([last_token]);
          callback_function(text);
        }
    
        this.update_kv_cache(outputs, feed);
        feed['input_ids'] = new ort.Tensor('int64', BigInt64Array.from([last_token]), [1, 1]);
        feed['position_ids'] = new ort.Tensor('int64', BigInt64Array.from([BigInt(seqlen)]), [1, 1]);
    }
  }
}

let llm = null;

async function generate(messages, max_output_tokens) {
  if (!llm) {
    return;
  }

  let state = "thinking"; // 'thinking' or 'answering'
  let startTime;
  let numTokens = 0;
  let tps;
  const token_callback_function = (tokens) => {
    startTime ??= performance.now();

    if (numTokens++ > 0) {
      tps = (numTokens / (performance.now() - startTime)) * 1000;
    }
    if (tokens[0] == llm.end_thinking_token_id) {
      state = "answering";
    }
  };
  const callback_function = (output) => {
    self.postMessage({
      status: "update",
      output,
      tps,
      numTokens,
      state,
    });
  };

  // Tell the main thread we are starting
  self.postMessage({ status: "start" });

  await llm.query(messages, max_output_tokens, callback_function, token_callback_function);

  // Send the output back to the main thread
  self.postMessage({
    status: "complete",
  });
}

async function load() {
  self.postMessage({
    status: "loading",
    data: "Loading model...",
  });

  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = '/models/';

  llm = new LLM();
  await llm.init();

  self.postMessage({
    status: "loading",
    data: "Compiling shaders and warming up model...",
  });

  const max_output_tokens = 2;
  await llm.query(["a"], max_output_tokens);

  self.postMessage({ status: "ready" });
}

async function check() {
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("WebGPU is not supported (no adapter found)");
    }
  } catch (e) {
    self.postMessage({
      status: "error",
      data: e.toString(),
    });
  }
}

// Listen for messages from the main thread
self.addEventListener("message", async (e) => {
  const { type, data } = e.data;

  switch (type) {
    case "check":
      check();
      break;

    case "load":
      load();
      break;

    case "generate":
      const max_output_tokens = 2048;
      generate(data, max_output_tokens);
      break;

    case "interrupt":
      break;

    case "reset":
      break;
  }
});
