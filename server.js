// server.js - OpenAI to NVIDIA NIM API Proxy
// Base: atualização do autor original (4 dias atrás)
// Adições: retry 504, token limiter, keep-alive Render, debug page, nossa lista de modelos

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NVIDIA_THIRD_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';

const REQUEST_TIMEOUT_MS = 600000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB
const MAX_TOKENS_LIMIT = 65536;

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config Validation ───────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NVIDIA_THIRD_API_KEY is required.');
  if (!CLIENT_AUTH_KEY) console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
}

validateConfig();

// ─── Model Mapping ───────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gpt-4':          'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-4o':         'meta/llama-3.3-70b-instruct',
  'claude-3-opus':  'openai/gpt-oss-120b',
  'claude-3-sonnet':'openai/gpt-oss-20b',
  'gemini-pro':     'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'mistral-nemo':   'mistralai/mistral-nemotron',
  'google-light':   'google/gemma-4-31b-it',
  'step-3.7-flash':       'stepfun-ai/step-3.7-flash',
  'glm-5.2':            'z-ai/glm-5.2'
};

// ─── Fallback Chain ──────────────────────────────────────────────────────────

const FALLBACK_MODELS = [
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'meta/llama-3.3-70b-instruct',
  'google/gemma-4-31b-it',
  'mistralai/mistral-nemotron'
];

// ─── Reasoning Subsystem ─────────────────────────────────────────────────────

class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    this.buffer += chunk;
    let content = '';
    let reasoning = '';

    while (true) {
      const targetTag = this.inThinking ? this.closeTag : this.openTag;
      const tagIndex = this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore = this.buffer.substring(0, tagIndex);
        if (this.inThinking) reasoning += textBefore;
        else content += textBefore;
        this.inThinking = !this.inThinking;
        this.buffer = this.buffer.substring(tagIndex + targetTag.length);
      } else {
        let partialLen = 0;
        const maxLen = Math.min(this.buffer.length, targetTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
          if (targetTag.startsWith(this.buffer.substring(this.buffer.length - i))) {
            partialLen = i;
            break;
          }
        }
        const textBefore = this.buffer.substring(0, this.buffer.length - partialLen);
        if (this.inThinking) reasoning += textBefore;
        else content += textBefore;
        this.buffer = this.buffer.substring(this.buffer.length - partialLen);
        break;
      }
    }
    return { content, reasoning };
  }

  flush() {
    let content = '';
    let reasoning = '';
    if (this.buffer) {
      if (this.inThinking) reasoning += this.buffer;
      else content += this.buffer;
      this.buffer = '';
    }
    return { content, reasoning };
  }
}

class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;
    if (
      model === 'qwen/qwen3.5-397b-a17b' ||
      model === 'nvidia/llama-3.3-nemotron-super-49b-v1.5' ||
      model === 'stepfun-ai/step-3.7-flash'
    ) {
      this.parser = new DelimiterParser('<think>', '</think>');
    }
  }

  processDelta(delta) {
    const normalizedDelta = { ...delta };
    let reasoning = normalizedDelta.reasoning || normalizedDelta.reasoning_content || '';
    let content = normalizedDelta.content || '';

    if (!reasoning && content && this.parser) {
      const parsed = this.parser.processChunk(content);
      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) normalizedDelta.content = content;
    else delete normalizedDelta.content;

    if (reasoning) normalizedDelta.reasoning = reasoning;
    else delete normalizedDelta.reasoning;

    delete normalizedDelta.reasoning_content;
    return normalizedDelta;
  }

  flush() {
    if (!this.parser) return { content: '', reasoning: '' };
    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;

  const message = choice.message || {};
  let reasoning = message.reasoning || message.reasoning_content || '';
  let content = message.content || '';

  if (!reasoning && content) {
    let parser = null;
    if (model === 'qwen/qwen3.5-397b-a17b' || model === 'nvidia/llama-3.3-nemotron-super-49b-v1.5') {
      parser = new DelimiterParser('<think>', '</think>');
    }
    if (parser) {
      const parsed = parser.processChunk(content);
      const flushed = parser.flush();
      content = (parsed.content || '') + (flushed.content || '');
      reasoning = (parsed.reasoning || '') + (flushed.reasoning || '');
    }
  }

  const newMessage = { ...message };
  if (content) newMessage.content = content;
  if (reasoning) newMessage.reasoning = reasoning;
  delete newMessage.reasoning_content;

  return { ...choice, message: newMessage };
}

function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  const effort = clientReasoningEffort;

  switch (model) {
    case 'nvidia/llama-3.3-nemotron-super-49b-v1.5': {
      if (!enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: true } };
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (hasTools) payload.chat_template_kwargs.force_nonempty_content = true;
      return payload;
    }

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (effort && ['low', 'medium', 'high'].includes(effort)) {
        return { reasoning_effort: effort };
      }
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'z-ai/glm-5.2': {
      const payload = {
        thinking: { type: enableThinking ? 'enabled' : 'disabled' }
      };
      if (enableThinking && effort) payload.reasoning_effort = effort;
      return payload;
    }

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: true } };
    }

    case 'stepfun-ai/step-3.7-flash': {
      if (enableThinking) return {};
      return { chat_template_kwargs: { thinking: false } };
    }

    default:
      return {};
  }
}

// ─── Middleware ──────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '100mb' }));

app.use((req, res, next) => {
  req.socket.setKeepAlive(true, 15000);
  req.socket.setTimeout(0);
  next();
});

// ─── Auth ────────────────────────────────────────────────────────────────────

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') return next();

  const token = extractBearerToken(req.headers.authorization);

  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: { message: 'Forbidden: Invalid or missing authentication', type: 'authentication_error', code: 403 }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: { message: 'Forbidden: Invalid authentication credentials', type: 'authentication_error', code: 403 }
    });
  }

  next();
});

// ─── Model Validation ────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set((response.data.data || []).map(m => m.id));
    const invalid = [];

    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      console.warn(`[VALIDATION] ${invalid.length} model(s) not found in catalog.`);
    } else {
      console.log('[VALIDATION] All models valid.');
    }

  } catch (err) {
    console.warn(`[VALIDATION] /v1/models check failed: ${err.message}. Skipping.`);
  }
}

// ─── Debug Store ─────────────────────────────────────────────────────────────

const debugStore = [];
const MAX_DEBUG_ENTRIES = 5;

function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

function saveDebugEntry(rawBody) {
  const messages = rawBody.messages || [];
  const entry = {
    timestamp: new Date().toISOString(),
    model_requested: rawBody.model,
    model_mapped: MODEL_MAPPING[rawBody.model] || 'fallback',
    temperature: rawBody.temperature,
    max_tokens: rawBody.max_tokens,
    stream: rawBody.stream,
    total_messages: messages.length,
    estimated_tokens: messages.reduce((sum, m) => sum + estimateTokens(JSON.stringify(m)), 0),
    messages: messages.map((m, i) => ({
      index: i,
      role: m.role,
      char_length: (m.content || '').length,
      estimated_tokens: estimateTokens(JSON.stringify(m)),
      content_preview:
        (m.content || '').length > 600
          ? (m.content || '').slice(0, 300) + '\n\n[... TRUNCADO ...]\n\n' + (m.content || '').slice(-300)
          : (m.content || '')
    }))
  };
  debugStore.unshift(entry);
  if (debugStore.length > MAX_DEBUG_ENTRIES) debugStore.pop();
}

function escapeHtml(text) {
  return (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Token Limiter ───────────────────────────────────────────────────────────

function limitMessagesByTokens(messages, maxTokens = 100000) {
  if (!messages || messages.length === 0) return messages;
  let totalTokens = 0;
  const keptMessages = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const tokens = estimateTokens(JSON.stringify(messages[i]));
    if (totalTokens + tokens <= maxTokens) {
      keptMessages.unshift(messages[i]);
      totalTokens += tokens;
    } else {
      break;
    }
  }
  return keptMessages;
}

// ─── Safe Write ──────────────────────────────────────────────────────────────

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Fallback Caller com Retry 504 ───────────────────────────────────────────

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools) {
  let lastError = null;

  for (const model of models) {
    let attempts = 0;
    const maxAttempts = 2;

    while (attempts < maxAttempts) {
      attempts++;
      try {
        const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);

        const response = await axios.post(
          `${NIM_API_BASE}/chat/completions`,
          { ...baseRequest, model, ...reasoningPayload },
          {
            headers: { Authorization: `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
            responseType: 'stream',
            timeout: REQUEST_TIMEOUT_MS
          }
        );

        console.log('[PROXY] Model used:', model);
        return { response, model };

      } catch (err) {
        lastError = err;
        const status = err.response?.status;

        if (status === 504 && attempts < maxAttempts) {
          console.warn(`[RETRY] Model ${model} returned 504, retrying (attempt ${attempts})...`);
          await new Promise(r => setTimeout(r, 2000));
          continue;
        }

        console.warn(
          `[FALLBACK] Model failed: ${model}`,
          status,
          err.response?.data?.error?.message || err.message
        );
        break;
      }
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Debug Page ──────────────────────────────────────────────────────────────

app.get('/debug', (req, res) => {
  if (debugStore.length === 0) {
    return res.send(`<html><body style="font-family:monospace;padding:20px;background:#111;color:#0f0"><h2>Debug - Nenhum request recebido ainda</h2></body></html>`);
  }

  const entryIndex = Math.min(parseInt(req.query.entry || '0'), debugStore.length - 1);
  const entry = debugStore[entryIndex];

  const messagesHTML = entry.messages.map((m) => `
<div style="border:1px solid #333;margin:8px 0;padding:12px;border-radius:6px;background:#1a1a1a">
  <div style="margin-bottom:8px">
    <span style="background:${m.role === 'system' ? '#4a3000' : m.role === 'user' ? '#003a4a' : '#1a3a00'};padding:2px 8px;border-radius:4px;font-size:12px">
      [${m.index}] ${m.role.toUpperCase()}
    </span>
    <span style="color:#888;font-size:12px;margin-left:10px">${m.char_length} chars · ~${m.estimated_tokens} tokens</span>
  </div>
  <pre style="white-space:pre-wrap;word-break:break-word;color:#ccc;font-size:13px;margin:0">${escapeHtml(m.content_preview)}</pre>
</div>`).join('');

  res.send(`
<html>
<head>
  <title>Proxy Debug</title>
  <style>
    body { font-family:monospace; padding:20px; background:#111; color:#eee }
    h2 { color:#0f0 }
    .stat { display:inline-block; background:#222; padding:6px 14px; border-radius:6px; margin:4px; font-size:13px }
    .stat span { color:#0f0; font-weight:bold }
  </style>
</head>
<body>
  <h2>Proxy Debug</h2>
  <div class="stat">Modelo pedido: <span>${entry.model_requested}</span></div>
  <div class="stat">Mapeado: <span>${entry.model_mapped}</span></div>
  <div class="stat">Tokens: <span>${entry.estimated_tokens.toLocaleString()}</span></div>
  <div class="stat">Stream: <span>${entry.stream ? 'sim' : 'não'}</span></div>
  <h3 style="color:#0af">Mensagens (${entry.total_messages})</h3>
  ${messagesHTML}
</body>
</html>`);
});

app.get('/debug/raw', (req, res) => {
  if (debugStore.length === 0) return res.json({ message: 'Nenhum request recebido ainda.' });
  res.json(debugStore[0]);
});

// ─── Routes ──────────────────────────────────────────────────────────────────

app.get('/health', (_, res) => {
  res.json({ status: 'ok', service: 'NVIDIA NIM Proxy', version: '2.2.0' });
});

app.get('/v1/models', (_, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map((m) => ({
      id: m,
      object: 'model',
      created: Date.now(),
      owned_by: 'nvidia-nim-proxy'
    }))
  });
});

// ─── Chat Completions ────────────────────────────────────────────────────────

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    saveDebugEntry(req.body);

    const primaryModel = MODEL_MAPPING[model] || 'nvidia/llama-3.3-nemotron-super-49b-v1.5';
    const modelChain = [primaryModel, ...FALLBACK_MODELS];
    const limitedMessages = limitMessagesByTokens(messages, 100000);

    const baseRequest = {
      messages: limitedMessages,
      temperature: temperature ?? 1.0,
      max_tokens: Math.min(max_tokens ?? 16384, MAX_TOKENS_LIMIT),
      stream: true
    };

    const { response, model: usedModel } = await callWithFallback(
      baseRequest,
      modelChain,
      ENABLE_THINKING_MODE,
      req.body.reasoning_effort,
      !!req.body.tools
    );

    upstreamStream = response.data;

    const inlineReasoning = req.headers['x-reasoning-format'] === 'inline';

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      const decoder = new StringDecoder('utf8');
      let sseBuffer = '';
      let fullReasoning = '';
      let fullContent = '';
      let lastData = null;
      let doneSent = false;
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) upstreamStream.removeAllListeners();
        req.removeAllListeners('close');
      };

      upstreamStream.on('data', chunk => {
        sseBuffer += decoder.write(chunk);

        if (sseBuffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith(':')) continue;
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) continue;

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta?.reasoning_content) fullReasoning += delta.reasoning_content;
            if (delta?.reasoning) fullReasoning += delta.reasoning;
            if (delta?.content) fullContent += delta.content;

            lastData = data;
          } catch (err) {
            console.warn('[STREAM] Skipped invalid JSON:', line.slice(0, 100));
          }
        }
      });

      upstreamStream.on('end', () => {
        sseBuffer += decoder.end();

        const finalContent = fullReasoning.length > 0
          ? `<think>${fullReasoning}</think>\n\n${fullContent}`
          : fullContent;

        if (lastData) {
          const finalChunk = {
            ...lastData,
            choices: [{
              index: 0,
              delta: { content: finalContent },
              finish_reason: lastData.choices?.[0]?.finish_reason || 'stop'
            }]
          };
          safeWrite(res, `data: ${JSON.stringify(finalChunk)}\n\n`);
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
          doneSent = true;
        }

        streamEndedCleanly = true;
        if (!res.writableEnded) res.end();
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        if (!res.writableEnded) {
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }
        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });

    } else {
      // Non-streaming
      let fullReasoning = '';
      let fullContent = '';
      let finishReason = 'stop';
      let usageData = null;
      let sseBuffer = '';
      const decoder = new StringDecoder('utf8');
      const normalizer = new StreamNormalizer(usedModel);

      upstreamStream.on('data', chunk => {
        sseBuffer += decoder.write(chunk);
        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ') || line.includes('[DONE]')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              const norm = normalizer.processDelta(delta);
              if (norm.reasoning) fullReasoning += norm.reasoning;
              if (norm.content) fullContent += norm.content;
            }
            if (data.choices?.[0]?.finish_reason) finishReason = data.choices[0].finish_reason;
            if (data.usage) usageData = data.usage;
          } catch {}
        }
      });

      upstreamStream.on('end', () => {
        const flushed = normalizer.flush();
        if (flushed.reasoning) fullReasoning += flushed.reasoning;
        if (flushed.content) fullContent += flushed.content;

        const finalContent = SHOW_REASONING && fullReasoning.length > 0
          ? `<think>${fullReasoning}</think>\n\n${fullContent}`
          : fullContent;

        res.json({
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{
            index: 0,
            message: { role: 'assistant', content: finalContent },
            finish_reason: finishReason
          }],
          usage: usageData ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        });
      });

      upstreamStream.on('error', err => {
        console.error('Error (non-stream):', err.message);
        if (!res.headersSent) res.status(500).json({ error: { message: err.message } });
      });
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.message || 'Internal server error', type: 'proxy_error', code: error.response?.status || 500 }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) upstreamStream.destroy();
  }
});

// ─── 404 ─────────────────────────────────────────────────────────────────────

app.use((req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not found`, type: 'invalid_request_error', code: 404 }
  });
});

// ─── Startup ─────────────────────────────────────────────────────────────────

const server = app.listen(PORT, () => {
  console.log(`✅ Proxy rodando na porta ${PORT}`);

  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });

  const RENDER_URL = process.env.RENDER_EXTERNAL_URL;
  if (RENDER_URL) {
    setInterval(() => {
      axios.get(`${RENDER_URL}/health`)
        .then(() => console.log('🏓 Keep-alive OK'))
        .catch(err => console.warn(`⚠️ Keep-alive falhou: ${err.message}`));
    }, 10 * 60 * 1000);
  }
});

server.setTimeout(0);
server.keepAliveTimeout = 620000;
server.headersTimeout = 630000;
