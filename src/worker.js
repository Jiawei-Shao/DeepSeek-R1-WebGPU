import { LLM } from "./llm"

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
