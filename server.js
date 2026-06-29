// server.js - OpenAI to NVIDIA NIM API Proxy

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ============================================================
// CONFIGURATION
// ============================================================

const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NVIDIA_SECOND_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';

const REQUEST_TIMEOUT_MS = 600000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');
if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ============================================================
// CONFIG VALIDATION
// ============================================================

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NVIDIA_SECOND_API_KEY is required.');
  if (!CLIENT_AUTH_KEY) console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
}

validateConfig();

// ============================================================
// MODEL MAPPING
// ============================================================

const MODEL_MAPPING = {
  'gpt-3.5-turbo':  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gpt-4':          'z-ai/glm-5.1',
  'gpt-4-turbo':    'moonshotai/kimi-k2.6',
  'gpt-4o':         'deepseek-ai/deepseek-v4-flash',
  'claude-3-opus':  'openai/gpt-oss-120b',
  'claude-3-sonnet':'openai/gpt-oss-20b',
  'gemini-pro':     'qwen/qwen3-next-80b-a3b-thinking',
  'mistral':        'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo':  'mistralai/mistral-medium-3.5-128b',
  'mistral-pro':    'mistralai/mistral-small-4-119b-2603',
  'mistral-fast':   'mistralai/ministral-14b-instruct-2512',
  'mistral-nemo':   'mistralai/mistral-nemotron',
  'google-light':   'google/gemma-4-31b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'google-lightest':'google/gemma-2-2b-it',
  'step':           'stepfun-ai/step-3.5-flash',
  'step-3.7':       'stepfun-ai/step-3.7-flash',
  'm2.7':           'minimaxai/minimax-m2.7'
};

// ============================================================
// FALLBACK CHAIN
// ============================================================

const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors());
app.use(express.json({ limit: '100mb' }));

app.use((req, res, next) => {
  req.socket.setKeepAlive(true, 15000);
  req.socket.setTimeout(0);
  next();
});

// ============================================================
// AUTH — timingSafeEqual previne timing attacks e bypass
// ============================================================

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
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);

  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ============================================================
// MODEL VALIDATION — usa /v1/models em vez de inferência
// ============================================================

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');

  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

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

// ============================================================
// DEBUG STORE
// ============================================================

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
    estimated_tokens: messages.reduce(
      (sum, m) => sum + estimateTokens(JSON.stringify(m)),
      0
    ),
    messages: messages.map((m, i) => ({
      index: i,
      role: m.role,
      char_length: (m.content || '').length,
      estimated_tokens: estimateTokens(JSON.stringify(m)),
      content_preview:
        (m.content || '').length > 600
          ? (m.content || '').slice(0, 300) +
            '\n\n[... TRUNCADO ...]\n\n' +
            (m.content || '').slice(-300)
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

// ============================================================
// FIX PARAGRAPHS
// ============================================================

function fixParagraphs(text) {
  if (!text) return text;

  let result = text;

  result = result.replace(/([*_"»])\n([*_"«*])/g, '$1\n\n$2');

  result = result.replace(
    /(\*\*"[^"]{1,60}[,.]?"\*\*)\n\n(\*[^*])/g,
    '$1 $2'
  );

  result = result.replace(
    /(\*[^*]{1,60}[,]\*)\n\n(\*\*")/g,
    '$1 $2'
  );

  result = result.replace(
    /(\*[^*]{1,40}[,]\*)\n\n(\*[A-Za-z])/g,
    '$1 $2'
  );

  result = result.replace(/\n{3,}/g, '\n\n');

  return result.trim();
}

// ============================================================
// SAFE WRITE — evita crashes em sockets fechados
// ============================================================

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

// ============================================================
// TOKEN LIMITER
// ============================================================

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

// ============================================================
// DEBUG PAGE
// ============================================================

app.get('/debug', (req, res) => {
  if (debugStore.length === 0) {
    return res.send(`<html><body style="font-family:monospace;padding:20px;background:#111;color:#0f0"><h2>Debug - Nenhum request recebido ainda</h2></body></html>`);
  }

  const entryIndex = Math.min(
    parseInt(req.query.entry || '0'),
    debugStore.length - 1
  );

  const entry = debugStore[entryIndex];

  const messagesHTML = entry.messages
    .map(
      (m) => `
<div style="border:1px solid #333;margin:8px 0;padding:12px;border-radius:6px;background:#1a1a1a">
  <div style="margin-bottom:8px">
    <span style="
      background:${
        m.role === 'system'
          ? '#4a3000'
          : m.role === 'user'
          ? '#003a4a'
          : '#1a3a00'
      };
      padding:2px 8px;
      border-radius:4px;
      font-size:12px
    ">
      [${m.index}] ${m.role.toUpperCase()}
    </span>
    <span style="color:#888;font-size:12px;margin-left:10px">
      ${m.char_length} chars · ~${m.estimated_tokens} tokens
    </span>
  </div>
  <pre style="
    white-space:pre-wrap;
    word-break:break-word;
    color:#ccc;
    font-size:13px;
    margin:0
  ">${escapeHtml(m.content_preview)}</pre>
</div>
`
    )
    .join('');

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
</html>
`);
});

app.get('/debug/raw', (req, res) => {
  if (debugStore.length === 0) {
    return res.json({ message: 'Nenhum request recebido ainda.' });
  }
  res.json(debugStore[0]);
});

// ============================================================
// ROUTES
// ============================================================

app.get('/health', (_, res) => {
  res.json({ status: 'ok', service: 'NVIDIA NIM Proxy', version: '2.1.0' });
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

// ============================================================
// FALLBACK CALLER
// ============================================================

async function callWithFallback(baseRequest, models) {
  let lastError = null;

  for (const model of models) {
    try {
      const response = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        { ...baseRequest, model },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: 'stream',
          timeout: REQUEST_TIMEOUT_MS
        }
      );

      console.log('[PROXY] Model used:', model);
      return { response, model };

    } catch (err) {
      lastError = err;
      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        err.response?.status,
        err.response?.data?.error?.message || err.message
      );
    }
  }

  throw lastError || new Error('All models failed');
}

// ============================================================
// CHAT COMPLETIONS
// ============================================================

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    saveDebugEntry(req.body);

    const primaryModel =
      MODEL_MAPPING[model] || 'nvidia/llama-3.3-nemotron-super-49b-v1.5';

    const modelChain = [primaryModel, ...FALLBACK_MODELS];

    const limitedMessages = limitMessagesByTokens(messages, 100000);

    const baseRequest = {
      messages: limitedMessages,
      temperature: temperature ?? 1.0,
      max_tokens: max_tokens ?? 16384,
      stream: true,
      ...(ENABLE_THINKING_MODE && {
        extra_body: {
          chat_template_kwargs: {
            enable_thinking: true,
            clear_thinking: false
          }
        }
      })
    };

    const { response, model: usedModel } =
      await callWithFallback(baseRequest, modelChain);

    upstreamStream = response.data;

    // ============================================================
    // STREAM MODE
    // ============================================================

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

      upstreamStream.on('data', (chunk) => {
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

            if (delta?.reasoning_content) {
              fullReasoning += delta.reasoning_content;
            }

            if (delta?.content) {
              fullContent += delta.content;
            }

            lastData = data;
          } catch (err) {
            console.warn('[STREAM] Skipped invalid JSON:', line.slice(0, 100));
          }
        }
      });

      upstreamStream.on('end', () => {
        sseBuffer += decoder.end();

        const fixedContent = fixParagraphs(fullContent);

        const finalContent =
          SHOW_REASONING && fullReasoning.length > 0
            ? `<think>${fullReasoning}</think>\n\n${fixedContent}`
            : fixedContent;

        if (lastData) {
          const finalChunk = {
            ...lastData,
            choices: [
              {
                index: 0,
                delta: { content: finalContent },
                finish_reason: lastData.choices?.[0]?.finish_reason || 'stop'
              }
            ]
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

      upstreamStream.on('error', (err) => {
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

    // ============================================================
    // NORMAL MODE
    // ============================================================

    } else {
      let fullReasoning = '';
      let fullContent = '';
      let finishReason = 'stop';
      let usageData = null;
      let sseBuffer = '';
      const decoder = new StringDecoder('utf8');

      upstreamStream.on('data', (chunk) => {
        sseBuffer += decoder.write(chunk);

        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith(':')) continue;
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) continue;

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta?.reasoning_content) {
              fullReasoning += delta.reasoning_content;
            }

            if (delta?.content) {
              fullContent += delta.content;
            }

            if (data.choices?.[0]?.finish_reason) {
              finishReason = data.choices[0].finish_reason;
            }

            if (data.usage) {
              usageData = data.usage;
            }
          } catch {}
        }
      });

      upstreamStream.on('end', () => {
        const fixedContent = fixParagraphs(fullContent);

        const finalContent =
          SHOW_REASONING && fullReasoning.length > 0
            ? `<think>${fullReasoning}</think>\n\n${fixedContent}`
            : fixedContent;

        res.json({
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [
            {
              index: 0,
              message: {
                role: 'assistant',
                content: finalContent
              },
              finish_reason: finishReason
            }
          ],
          usage:
            usageData ?? {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0
            }
        });
      });

      upstreamStream.on('error', (err) => {
        console.error('Error (non-stream):', err.message);
        if (!res.headersSent) {
          res.status(500).json({ error: { message: err.message } });
        }
      });
    }

  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message || 'Internal server error',
          type: 'proxy_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

// ============================================================
// DIAGNÓSTICO
// ============================================================

app.post('/v1/diagnose', async (req, res) => {
  const nimModel = 'nvidia/llama-3.3-nemotron-super-49b-v1.5';

  const nimRequest = {
    model: nimModel,
    messages: [{ role: 'user', content: 'Olá, tudo bem?' }],
    temperature: 1.0,
    max_tokens: 500,
    stream: true,
    extra_body: {
      chat_template_kwargs: {
        enable_thinking: true,
        clear_thinking: false
      }
    }
  };

  const response = await axios.post(
    `${NIM_API_BASE}/chat/completions`,
    nimRequest,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: 'stream',
      timeout: 60000
    }
  );

  const chunks = [];
  let sseBuffer = '';

  response.data.on('data', (chunk) => {
    sseBuffer += chunk.toString();
    const lines = sseBuffer.split('\n');
    sseBuffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      if (line.includes('[DONE]')) continue;
      try {
        const data = JSON.parse(line.slice(6));
        const delta = data.choices?.[0]?.delta;
        if (delta) {
          chunks.push({
            has_content: !!delta.content,
            has_reasoning: !!delta.reasoning_content,
            content_preview: (delta.content || '').slice(0, 80),
            reasoning_preview: (delta.reasoning_content || '').slice(0, 80)
          });
        }
      } catch {}
    }
  });

  response.data.on('end', () => {
    res.json({
      total_chunks: chunks.length,
      chunks_with_reasoning: chunks.filter(c => c.has_reasoning).length,
      chunks_with_content: chunks.filter(c => c.has_content).length,
      first_5_chunks: chunks.slice(0, 5),
      last_5_chunks: chunks.slice(-5)
    });
  });
});

// ============================================================
// 404 FALLBACK
// ============================================================

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ============================================================
// START SERVER
// ============================================================

const server = app.listen(PORT, () => {
  console.log(`✅ Proxy rodando na porta ${PORT}`);

  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });

  const RENDER_URL = process.env.RENDER_EXTERNAL_URL;

  if (RENDER_URL) {
    setInterval(() => {
      axios
        .get(`${RENDER_URL}/health`)
        .then(() => console.log('🏓 Keep-alive OK'))
        .catch((err) =>
          console.warn(`⚠️ Keep-alive falhou: ${err.message}`)
        );
    }, 10 * 60 * 1000);
  }
});

server.setTimeout(0);
server.keepAliveTimeout = 620000;
server.headersTimeout = 630000;
