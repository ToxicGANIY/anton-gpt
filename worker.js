import { pipeline, TextStreamer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

const MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct";
const SYSTEM = "Ты глупый, самоуверенный бот, который часто путается и отвечает невпопад. " +
  "Отвечай по-русски, коротко и с абсурдным юмором. Это шуточный режим.";

const SERIOUS_SYSTEM = "Ты обычный полезный чатбот и собеседник. Внимательно реагируй именно на смысл последнего сообщения пользователя и учитывай контекст переписки. " +
  "Отвечай на вопрос, выполняй просьбу или поддерживай разговор в соответствии с тем, что написал пользователь; не отвечай шаблонно и не меняй тему. " +
  "Если пользователь здоровается — поздоровайся; если делится мыслью или переживанием — отреагируй на это; если просит объяснить, придумать или помочь — сделай это. " +
  "Пиши естественно и по-русски. Будь серьёзен, доброжелателен и по делу, но выбирай длину ответа по задаче: не обрывай сложные объяснения и не расписывай простые. " +
  "Не выдумывай факты и не утверждай, что сделал то, чего не делал; если не уверен или не хватает данных, прямо скажи об этом или задай уточняющий вопрос.";

let gen = null;
let backend = "";
let messageCount = 0;
const conversations = { serious: [], silly: [] };

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

  messageCount++;
  // Цикл по 10 ответов: первые 5 — глупые, следующие 5 — серьёзные.
  const isSerious = Math.floor((messageCount - 1) / 5) % 2 === 1;
  const systemPrompt = isSerious ? SERIOUS_SYSTEM : SYSTEM;

  postMessage({ type: "mode", serious: isSerious, count: messageCount });

  let answer = "";
  const streamer = new TextStreamer(gen.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (t) => {
      answer += t;
      postMessage({ type: "token", text: t });
    },
  });
  try {
    const modeConversation = conversations[isSerious ? "serious" : "silly"];
    const messages = [
      { role: "system", content: systemPrompt },
      ...modeConversation,
      { role: "user", content: text },
    ];
    await gen(
      messages,
      { max_new_tokens: isSerious ? 160 : 60, do_sample: true, temperature: isSerious ? 0.3 : temperature, streamer }
    );
    modeConversation.push({ role: "user", content: text }, { role: "assistant", content: answer.trim() });
    // У каждого режима своя история; храним последние 6 пар реплик.
    if (modeConversation.length > 12) modeConversation.splice(0, modeConversation.length - 12);
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