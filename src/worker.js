import {
  AutoTokenizer,
  env,
} from "@huggingface/transformers";
import ort from 'onnxruntime-web/webgpu'

class LLM {
  model_id = "DeepSeek-R1-Distill-Qwen-1.5B-ONNX";

  tokenizer = undefined;
  inferenceSession = undefined;

  kv_dims = [];
  num_hidden_layers = 0;

  eos = 0n;
  end_thinking_token_id = 0;

  async init() {
    ort.env.wasm.wasmPaths = '/dist/';
    ort.env.wasm.numThreads = 1;

    this.tokenizer = await AutoTokenizer.from_pretrained(this.model_id);

    // 151648: <think>
    // 151649: </think>
    const [END_THINKING_TOKEN_ID] = llm.tokenizer.encode(
      "</think>",
      { add_special_tokens: false },
    );
    this.end_thinking_token_id = END_THINKING_TOKEN_ID;

    const modelBytes = await this.fetchAndCache(`/models/${this.model_id}/onnx/model_q4f16.onnx`);
    let modelSize = modelBytes.byteLength;
    console.log(`${Math.round(modelSize / 1024 / 1024)} MB`);
    const inferenceSessionOptions = {
      executionProviders: ["webgpu"],
      preferredOutputLocation: {},
    };
    const jsonBytes = await this.fetchAndCache(`/models/${this.model_id}/config.json`);
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
    //console.log('Create session success!');

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

  async fetchAndCache(url) {
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            //console.log(`${url} (network)`);
            const buffer = await fetch(url).then(response => response.arrayBuffer());
            try {
                await cache.put(url, new Response(buffer));
            } catch (error) {
                console.error(error);   
            }
            return buffer;
        }
        //console.log(`${url} (cached)`);
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        console.log(`can't fetch ${url}`);
        throw error;
    }
  }

  async query(messages, callback_function = null, token_callback_function = null) {
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

    //console.log('Start inferencing.')
    let last_token = 0n;
    const kMaxOutputTokens = 2048;
    while (last_token != this.eos && seqlen < kMaxOutputTokens) {
        
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
    //console.log('Inferencing completed!')
  }
}

let llm = null;

async function generate(messages) {
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

  await llm.query(messages, callback_function, token_callback_function);

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

  llm.query(["a"]);

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
      generate(data);
      break;

    case "interrupt":
      break;

    case "reset":
      break;
  }
});
