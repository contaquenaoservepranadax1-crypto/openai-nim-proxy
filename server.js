// server.js - OpenAI to NVIDIA NIM API Proxy (Versão Híbrida by Pedro & GPT-5)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' })); // 🚀 buffer grande p/ conversas longas

// NVIDIA NIM API config
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Configurações de controle
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = true; // ✅ reativado para respostas mais completas

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.3-70b-instruct',
  'gpt-4': 'nvidia/llama-3.1-nemotron-70b-instruct',
  'gpt-4-turbo': 'qwen/qwen2.5-72b-instruct',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1-terminus', // 🧠 Terminus principal
  'claude-3-opus': 'meta/llama-3.1-405b-instruct',
  'claude-3-sonnet': 'meta/llama-3.3-70b-instruct',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-ultra-253b-v1'
};

// Estimativa de tokens (1 token ≈ 4 caracteres)
function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

// 🧠 Limite adaptativo de histórico (mantém coerência e profundidade)
function limitMessagesByTokens(messages, maxTokens = 8000) {
  if (!messages || messages.length === 0) return messages;

  let totalTokens = 0;
  const keptMessages = [];

  // percorre de trás pra frente, mantendo máximo possível
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    const tokens = estimateTokens(JSON.stringify(msg));
    if (totalTokens + tokens <= maxTokens) {
      keptMessages.unshift(msg);
      totalTokens += tokens;
    } else break;
  }

  return keptMessages;
}

// Health check
app.get('/health', (_, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models
app.get('/v1/models', (_, res) => {
  const models = Object.keys(MODEL_MAPPING).map(m => ({
    id: m,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Chat completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    let nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';
    const limitedMessages = limitMessagesByTokens(messages, 8000);

    // Request para NIM
    const nimRequest = {
      model: nimModel,
      messages: limitedMessages,
      temperature: temperature ?? 0.8, // 💫 mais criativo
      max_tokens: max_tokens ?? 16384, // 💬 respostas longas
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: !!stream
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    // STREAM MODE
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              if (!SHOW_REASONING) delete delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', err => {
        console.error('Stream error:', err.message);
        res.end();
      });
    }

    // NORMAL MODE
    else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => ({
          index: choice.index,
          message: {
            role: choice.message.role,
            content: SHOW_REASONING && choice.message.reasoning_content
              ? `<think>\n${choice.message.reasoning_content}\n</think>\n\n${choice.message.content}`
              : choice.message.content
          },
          finish_reason: choice.finish_reason
        })),
        usage: response.data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'proxy_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Fallback
app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} not found`, code: 404 }
  });
});

app.listen(PORT, () => {
  console.log(`✅ Proxy rodando na porta ${PORT}`);
  console.log(`🌐 Health: http://localhost:${PORT}/health`);
  console.log(`🧠 Thinking: ${ENABLE_THINKING_MODE ? 'Ativado' : 'Desativado'}`);
});
