import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

const MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct";
const SYSTEM = "Ты очень глупый и самоуверенный бот. Ты всё путаешь, придумываешь несуществующие факты, " +
  "отвечаешь невпопад и совершенно серьёзно несёшь абсурд. Отвечай коротко, по-русски.";

let gen = null;
let backend = "";

async function load(device, dtype) {
  return pipeline("text-generation", MODEL, {
    device,
    dtype,
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        postMessage({
          type: "progress",
          percent: Math.round((p.loaded / p.total) * 100),
          device
        });
      }
    },
  });
}

async function init(useGpu) {
  try {
    if (navigator.gpu && useGpu) {
      try {
        gen = await load("webgpu", "q4");
        backend = "видеокарта";
      } catch (e) {
        console.warn("WebGPU не вышел, перехожу на процессор:", e);
      }
    }
    if (!gen) {
      gen = await load("wasm", "q8");
      backend = "процессор";
    }
    postMessage({ type: "ready", backend });
  } catch (e) {
    postMessage({ type: "error", message: e.message || String(e) });
  }
}

async function generate(text, temperature) {
  if (!gen) return;
  const streamer = new TextStreamer(gen.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t) => {
      postMessage({ type: "token", text: t });
    },
  });
  try {
    await gen(
      [{ role: "system", content: SYSTEM }, { role: "user", content: text }],
      { max_new_tokens: 60, do_sample: true, temperature, streamer }
    );
    postMessage({ type: "done" });
  } catch (e) {
    postMessage({ type: "error", message: e.message || String(e) });
  }
}

self.onmessage = (e) => {
  const { type, payload } = e.data;
  if (type === "init") init(payload.useGpu);
  else if (type === "generate") generate(payload.text, payload.temperature);
};