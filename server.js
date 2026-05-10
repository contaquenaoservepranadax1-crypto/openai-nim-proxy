// server.js - OpenAI to NVIDIA NIM API Proxy (THINKING ENABLED)

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// ============================================================
// Middleware
// ============================================================

app.use(cors());
app.use(express.json({ limit: '100mb' }));

app.use((req, res, next) => {
  req.socket.setKeepAlive(true, 15000);
  req.socket.setTimeout(0);
  next();
});

// ============================================================
// NVIDIA CONFIG
// ============================================================

const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NVIDIA_SECOND_API_KEY;

// ============================================================
// MODEL MAPPING
// ============================================================

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'z-ai/glm4.7',
  'gpt-4': 'z-ai/glm-5.1',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

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
    model_mapped:
      MODEL_MAPPING[rawBody.model] || 'meta/llama-3.1-70b-instruct',

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

  if (debugStore.length > MAX_DEBUG_ENTRIES) {
    debugStore.pop();
  }
}

function escapeHtml(text) {
  return (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
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
// TOKEN LIMITER
// ============================================================

function limitMessagesByTokens(messages, maxTokens = 100000) {
  if (!messages || messages.length === 0) {
    return messages;
  }

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
// HELPER: detecta se o modelo é GLM 5.1
// ============================================================

function isGLM51(nimModel) {
  return nimModel === 'z-ai/glm-5.1';
}

// ============================================================
// ROUTES
// ============================================================

app.get('/health', (_, res) => {
  res.json({ status: 'ok', service: 'NVIDIA NIM Proxy' });
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
// CHAT COMPLETIONS
// ============================================================

app.post('/v1/chat/completions', async (req, res) => {
  const { model, messages, temperature, max_tokens, stream } = req.body;

  saveDebugEntry(req.body);

  const nimModel =
    MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';

  const limitedMessages = limitMessagesByTokens(messages, 100000);

  const chatTemplateKwargs = isGLM51(nimModel)
    ? {
        thinking: true,
        clear_thinking: true,
        do_sample: true,
        enable_thinking: true
      }
    : {
        thinking: true
      };

  const nimRequest = {
    model: nimModel,
    messages: limitedMessages,
    temperature: temperature ?? 1.0,
    max_tokens: max_tokens ?? 16384,
    stream: true,
    extra_body: {
      chat_template_kwargs: chatTemplateKwargs
    }
  };

  try {
    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: 'stream',
        timeout: 600000
      }
    );

    // ============================================================
    // STREAM MODE
    // ============================================================

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      let sseBuffer = '';

      // Estado da acumulação do reasoning
      let reasoningBuffer = '';
      let reasoningClosed = false;
      let firstContentSent = false;

      response.data.on('data', (chunk) => {
        sseBuffer += chunk.toString();

        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith(':')) continue;
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (!delta) {
              res.write(`data: ${JSON.stringify(data)}\n\n`);
              continue;
            }

            const hasReasoning = typeof delta.reasoning_content === 'string' && delta.reasoning_content.length > 0;
            const hasContent = typeof delta.content === 'string' && delta.content.length > 0;

            // --- Chunk só com reasoning_content ---
            if (hasReasoning && !hasContent) {
              reasoningBuffer += delta.reasoning_content;

              // Manda o reasoning acumulado como <think> ainda aberto
              // Substitui delta para mandar só o pedaço novo como texto dentro da tag
              if (reasoningBuffer === delta.reasoning_content) {
                // Primeiro chunk de reasoning: abre a tag
                delta.content = `<think>${delta.reasoning_content}`;
              } else {
                // Chunks seguintes: só o texto novo (tag já foi aberta)
                delta.content = delta.reasoning_content;
              }
              delete delta.reasoning_content;

              res.write(`data: ${JSON.stringify(data)}\n\n`);
              continue;
            }

            // --- Primeiro chunk com content real (reasoning acabou) ---
            if (hasContent && !reasoningClosed && reasoningBuffer.length > 0) {
              reasoningClosed = true;
              firstContentSent = true;

              // Fecha a tag <think> e começa o content normal
              delta.content = `</think>\n\n${delta.content}`;
              delete delta.reasoning_content;

              res.write(`data: ${JSON.stringify(data)}\n\n`);
              continue;
            }

            // --- Chunks normais de content ---
            if (hasContent) {
              delete delta.reasoning_content;
              res.write(`data: ${JSON.stringify(data)}\n\n`);
              continue;
            }

            // --- Qualquer outro chunk (finish_reason, usage, etc.) ---
            delete delta.reasoning_content;
            res.write(`data: ${JSON.stringify(data)}\n\n`);

          } catch (err) {
            console.error('Chunk parse error:', err.message);
          }
        }
      });

      response.data.on('end', () => {
        // Se o stream terminou mas a tag <think> nunca foi fechada
        // (modelo mandou só reasoning sem content), fecha aqui
        if (reasoningBuffer.length > 0 && !reasoningClosed) {
          const closeChunk = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [
              {
                index: 0,
                delta: { content: '</think>' },
                finish_reason: null
              }
            ]
          };
          res.write(`data: ${JSON.stringify(closeChunk)}\n\n`);
        }

        if (!res.writableEnded) res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        if (!res.writableEnded) res.end();
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

      response.data.on('data', (chunk) => {
        sseBuffer += chunk.toString();

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

      response.data.on('end', () => {
        // Monta o content final com thinking na frente se existir
        const finalContent = fullReasoning.length > 0
          ? `<think>${fullReasoning}</think>\n\n${fullContent}`
          : fullContent;

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

      response.data.on('error', (err) => {
        console.error('Error (non-stream):', err.message);
        if (!res.headersSent) {
          res.status(500).json({ error: { message: err.message } });
        }
      });
    }
  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);

    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message || 'Internal server error',
          type: 'proxy_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      res.end();
    }
  }
});
// ============================================================
// DIAGNÓSTICO - ver raw chunks da NVIDIA
// ============================================================

app.post('/v1/diagnose', async (req, res) => {
  const nimModel = 'z-ai/glm-5.1';

  const nimRequest = {
    model: nimModel,
    messages: [{ role: 'user', content: 'Olá, tudo bem?' }],
    temperature: 1.0,
    max_tokens: 500,
    stream: true,
    extra_body: {
      chat_template_kwargs: {
        thinking: true,
        clear_thinking: true,
        do_sample: true,
        enable_thinking: true
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
// FALLBACK
// ============================================================

app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      code: 404
    }
  });
});

// ============================================================
// START SERVER
// ============================================================

const server = app.listen(PORT, () => {
  console.log(`✅ Proxy rodando na porta ${PORT}`);

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
