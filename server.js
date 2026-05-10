// server.js - OpenAI to NVIDIA NIM API Proxy
// THINKING ENABLED + JANITOR FORMAT FIX FINAL

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
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY =
  process.env.NVIDIA_SECOND_API_KEY;

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

  // thinking model
  'gemini-pro':
    'qwen/qwen3-next-80b-a3b-thinking'
};

// ============================================================
// THINKING SUPPORT
// ============================================================

function modelSupportsThinking(modelName) {
  const THINKING_MODELS = [
    'z-ai/glm-5.1',
    'qwen/qwen3-next-80b-a3b-thinking',
    'deepseek-ai/deepseek-v4-pro'
  ];

  return THINKING_MODELS.includes(modelName);
}

// ============================================================
// FORMAT FIX
// ============================================================

function improveFormatting(text) {
  if (!text) return '';

  let formatted = text;

  // remove carriage return
  formatted = formatted.replace(/\r/g, '');

  // junta palavras grudadas
  formatted = formatted.replace(
    /([a-z,])([A-Z])/g,
    '$1 $2'
  );

  formatted = formatted.replace(
    /([a-z])([A-Z][a-z])/g,
    '$1 $2'
  );

  // remove espaços duplicados
  formatted = formatted.replace(
    /[ \t]{2,}/g,
    ' '
  );

  // remove line breaks absurdos
  formatted = formatted.replace(
    /\n{3,}/g,
    '\n\n'
  );

  // ============================================================
  // FORMATAÇÃO RP
  // ============================================================

  // ação -> fala
  formatted = formatted.replace(
    /\*\s*"/g,
    '*\n\n"'
  );

  // fala -> ação
  formatted = formatted.replace(
    /"\s*\*/g,
    '"\n\n*'
  );

  // ação -> ação
  formatted = formatted.replace(
    /\*\s*\*/g,
    '*\n\n*'
  );

  // fala -> fala
  formatted = formatted.replace(
    /"\s*"/g,
    '"\n\n"'
  );

  // remove linhas gigantes
  formatted = formatted.replace(
    /\n{3,}/g,
    '\n\n'
  );

  return formatted.trim();
}

// ============================================================
// THINKING EXTRACTOR
// ============================================================

function extractThinking(text) {
  if (!text) {
    return {
      content: '',
      reasoning: ''
    };
  }

  let reasoning = '';

  const thinkRegex =
    /<think>([\s\S]*?)<\/think>/gi;

  const cleanContent = text.replace(
    thinkRegex,
    (_, thinkContent) => {
      reasoning += thinkContent.trim() + '\n';
      return '';
    }
  );

  return {
    content: cleanContent.trim(),
    reasoning: reasoning.trim()
  };
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

    model_mapped:
      MODEL_MAPPING[rawBody.model] ||
      'meta/llama-3.1-70b-instruct',

    temperature: rawBody.temperature,
    max_tokens: rawBody.max_tokens,
    stream: rawBody.stream,

    total_messages: messages.length,

    estimated_tokens: messages.reduce(
      (sum, m) =>
        sum + estimateTokens(JSON.stringify(m)),
      0
    ),

    messages: messages.map((m, i) => ({
      index: i,
      role: m.role,

      char_length: (m.content || '').length,

      estimated_tokens: estimateTokens(
        JSON.stringify(m)
      ),

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

// ============================================================
// TOKEN LIMITER
// ============================================================

function limitMessagesByTokens(
  messages,
  maxTokens = 100000
) {
  if (!messages || messages.length === 0) {
    return messages;
  }

  let totalTokens = 0;

  const keptMessages = [];

  for (let i = messages.length - 1; i >= 0; i--) {

    const tokens = estimateTokens(
      JSON.stringify(messages[i])
    );

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
// ROUTES
// ============================================================

app.get('/health', (_, res) => {
  res.json({
    status: 'ok',
    service: 'NVIDIA NIM Proxy'
  });
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

  const {
    model,
    messages,
    temperature,
    max_tokens,
    stream
  } = req.body;

  saveDebugEntry(req.body);

  const nimModel =
    MODEL_MAPPING[model] ||
    'meta/llama-3.1-70b-instruct';

  const limitedMessages =
    limitMessagesByTokens(messages, 100000);

  const nimRequest = {
    model: nimModel,

    messages: limitedMessages,

    temperature: temperature ?? 1.0,

    max_tokens: max_tokens ?? 16384,

    stream: true
  };

  // ============================================================
  // ENABLE THINKING
  // ============================================================

  if (modelSupportsThinking(nimModel)) {

    nimRequest.extra_body = {
      chat_template_kwargs: {
        thinking: true
      }
    };
  }

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

      res.setHeader(
        'Content-Type',
        'text/event-stream'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      res.setHeader(
        'X-Accel-Buffering',
        'no'
      );

      let sseBuffer = '';

      response.data.on('data', (chunk) => {

        sseBuffer += chunk.toString();

        const lines = sseBuffer.split('\n');

        sseBuffer = lines.pop() || '';

        for (const line of lines) {

          if (line.startsWith(':')) continue;

          if (!line.startsWith('data: '))
            continue;

          if (line.includes('[DONE]')) {

            res.write('data: [DONE]\n\n');

            continue;
          }

          try {

            const data = JSON.parse(
              line.slice(6)
            );

            const delta =
              data.choices?.[0]?.delta;

            if (delta?.content) {

              const extracted =
                extractThinking(delta.content);

              // thinking box
              if (extracted.reasoning) {

                delta.reasoning_content =
                  extracted.reasoning;

                delta.reasoning =
                  extracted.reasoning;

                delta.thinking =
                  extracted.reasoning;
              }

              // NÃO formatar stream
              delta.content =
                extracted.content;
            }

            res.write(
              `data: ${JSON.stringify(data)}\n\n`
            );

          } catch (err) {

            console.error(
              'Chunk parse error:',
              err.message
            );
          }
        }
      });

      response.data.on('end', () => {

        if (!res.writableEnded) {
          res.end();
        }
      });

      response.data.on('error', (err) => {

        console.error(
          'Stream error:',
          err.message
        );

        if (!res.writableEnded) {
          res.end();
        }
      });

    // ============================================================
    // NORMAL MODE
    // ============================================================

    } else {

      let fullContent = '';
      let globalReasoning = '';

      let finishReason = 'stop';
      let usageData = null;

      let sseBuffer = '';

      response.data.on('data', (chunk) => {

        sseBuffer += chunk.toString();

        const lines = sseBuffer.split('\n');

        sseBuffer = lines.pop() || '';

        for (const line of lines) {

          if (line.startsWith(':')) continue;

          if (!line.startsWith('data: '))
            continue;

          if (line.includes('[DONE]'))
            continue;

          try {

            const data = JSON.parse(
              line.slice(6)
            );

            const delta =
              data.choices?.[0]?.delta;

            if (delta?.content) {

              const extracted =
                extractThinking(delta.content);

              if (extracted.reasoning) {

                globalReasoning +=
                  extracted.reasoning + '\n';
              }

              fullContent +=
                extracted.content;
            }

            if (
              data.choices?.[0]?.finish_reason
            ) {

              finishReason =
                data.choices[0]
                  .finish_reason;
            }

            if (data.usage) {
              usageData = data.usage;
            }

          } catch {}
        }
      });

      response.data.on('end', () => {

        res.json({

          id: `chatcmpl-${Date.now()}`,

          object: 'chat.completion',

          created: Math.floor(
            Date.now() / 1000
          ),

          model,

          choices: [
            {
              index: 0,

              message: {
                role: 'assistant',

                content: improveFormatting(
                  fullContent
                ),

                reasoning_content:
                  globalReasoning.trim(),

                reasoning:
                  globalReasoning.trim(),

                thinking:
                  globalReasoning.trim()
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

        console.error(
          'Error (non-stream):',
          err.message
        );

        if (!res.headersSent) {

          res.status(500).json({
            error: {
              message: err.message
            }
          });
        }
      });
    }

  } catch (error) {

    console.error(
      'Proxy error:',
      error.response?.data ||
      error.message
    );

    if (!res.headersSent) {

      res.status(
        error.response?.status || 500
      ).json({
        error: {
          message:
            error.message ||
            'Internal server error',

          type: 'proxy_error',

          code:
            error.response?.status || 500
        }
      });

    } else if (!res.writableEnded) {

      res.end();
    }
  }
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

  console.log(
    `✅ Proxy rodando na porta ${PORT}`
  );

  const RENDER_URL =
    process.env.RENDER_EXTERNAL_URL;

  if (RENDER_URL) {

    setInterval(() => {

      axios
        .get(`${RENDER_URL}/health`)

        .then(() =>
          console.log('🏓 Keep-alive OK')
        )

        .catch((err) =>
          console.warn(
            `⚠️ Keep-alive falhou: ${err.message}`
          )
        );

    }, 10 * 60 * 1000);
  }
});

server.setTimeout(0);

server.keepAliveTimeout = 620000;

server.headersTimeout = 630000;
