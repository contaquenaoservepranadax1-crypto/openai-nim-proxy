
// server.js - OpenAI to NVIDIA NIM API Proxy (OTIMIZADO PARA VELOCIDADE)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));

// NVIDIA NIM API config
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NVIDIA_SECOND_API_KEY;

// Configurações de controle
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Model mapping - ATUALIZADO com DeepSeek V3.1 base (funciona!)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'moonshotai/kimi-k2.5',  // ⚡ Teste tmb
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',              // 💭 Emocional
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.2',                  // 🧠 DeepSeek V3.2
  'gpt-4o': 'deepseek-ai/deepseek-v3.1-terminus',                       // 🧠 DeepSeek V3.1 Terminus 
  'gpt-4o-nemotron': 'nvidia/llama-3.1-nemotron-ultra-253b-v1', // ⚡ Backup rápido
  'claude-3-opus': 'z-ai/glm5',         // 🤔 Teste
  'gemini-pro': 'minimaxai/minimax-m2.5'      // Outro teste 
};

// ============================================================
// 🔍 DEBUG STORE - guarda os últimos 5 requests recebidos
// ============================================================
const debugStore = [];
const MAX_DEBUG_ENTRIES = 5;

// Estimativa de tokens (simplificada)
function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

function saveDebugEntry(rawBody) {
  const messages = rawBody.messages || [];

  const entry = {
    timestamp: new Date().toISOString(),
    model_requested: rawBody.model,
    model_mapped: MODEL_MAPPING[rawBody.model] || 'meta/llama-3.1-70b-instruct',
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
      content_preview: (m.content || '').length > 600
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

// ============================================================
// 🔍 ENDPOINT DE DEBUG - abre no browser
// ============================================================
app.get('/debug', (req, res) => {
  if (debugStore.length === 0) {
    return res.send(`
      <html><body style="font-family:monospace;padding:20px;background:#111;color:#0f0">
        <h2>🔍 Debug — Nenhum request recebido ainda</h2>
        <p>Faça uma mensagem no JanitorAI e recarregue esta página.</p>
      </body></html>
    `);
  }

  const entryIndex = Math.min(parseInt(req.query.entry || '0'), debugStore.length - 1);
  const entry = debugStore[entryIndex];

  const messagesHTML = entry.messages.map(m => `
    <div style="border:1px solid #333;margin:8px 0;padding:12px;border-radius:6px;background:#1a1a1a">
      <div style="margin-bottom:8px">
        <span style="background:${m.role === 'system' ? '#4a3000' : m.role === 'user' ? '#003a4a' : '#1a3a00'};padding:2px 8px;border-radius:4px;font-size:12px">
          [${m.index}] ${m.role.toUpperCase()}
        </span>
        <span style="color:#888;font-size:12px;margin-left:10px">
          ${m.char_length} chars · ~${m.estimated_tokens} tokens
        </span>
      </div>
      <pre style="white-space:pre-wrap;word-break:break-word;color:#ccc;font-size:13px;margin:0">${escapeHtml(m.content_preview)}</pre>
    </div>
  `).join('');

  const allEntriesNav = debugStore.map((e, i) => `
    <a href="/debug?entry=${i}" style="color:${i === entryIndex ? '#0f0' : '#666'};margin-right:15px;text-decoration:none;font-size:12px">
      ${i === entryIndex ? '▶ ' : ''}[${i}] ${e.timestamp} — ${e.total_messages} msgs
    </a>
  `).join('<br>');

  res.send(`
    <html>
    <head>
      <title>🔍 Proxy Debug</title>
      <meta charset="utf-8">
      <style>
        body { font-family: monospace; padding: 20px; background: #111; color: #eee; }
        h2 { color: #0f0; }
        .stat { display: inline-block; background: #222; padding: 6px 14px; border-radius: 6px; margin: 4px; font-size: 13px; }
        .stat span { color: #0f0; font-weight: bold; }
        .nav { background: #1a1a1a; padding: 12px; border-radius: 6px; margin-bottom: 20px; font-size: 12px; line-height: 2; }
      </style>
    </head>
    <body>
      <h2>🔍 Proxy Debug</h2>

      <div class="nav">
        <b style="color:#888">Histórico (últimos ${MAX_DEBUG_ENTRIES} requests):</b><br>
        ${allEntriesNav}
      </div>

      <div style="margin-bottom:16px">
        <div class="stat">🕐 <span>${entry.timestamp}</span></div>
        <div class="stat">🤖 Modelo pedido: <span>${entry.model_requested}</span></div>
        <div class="stat">🔀 Mapeado para: <span>${entry.model_mapped}</span></div>
        <div class="stat">📨 Total mensagens: <span>${entry.total_messages}</span></div>
        <div class="stat">🔢 Tokens estimados: <span>${entry.estimated_tokens.toLocaleString()}</span></div>
        <div class="stat">🌡️ Temperature: <span>${entry.temperature ?? 'default'}</span></div>
        <div class="stat">📏 Max tokens: <span>${entry.max_tokens ?? 'default'}</span></div>
        <div class="stat">📡 Stream: <span>${entry.stream ? 'sim' : 'não'}</span></div>
      </div>

      <h3 style="color:#0af">📋 Mensagens (${entry.total_messages} total) — conteúdos longos truncados</h3>
      ${messagesHTML}

      <br>
      <button onclick="location.reload()" style="background:#0f0;color:#000;border:none;padding:10px 20px;border-radius:6px;cursor:pointer;font-weight:bold;margin-right:10px">
        🔄 Atualizar
      </button>
      <a href="/debug/raw" style="background:#333;color:#eee;padding:10px 20px;border-radius:6px;text-decoration:none">
        📄 Ver JSON bruto
      </a>
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

// Limite adaptativo de histórico (AUMENTADO)
function limitMessagesByTokens(messages, maxTokens = 100000) {
  if (!messages || messages.length === 0) return messages;

  let totalTokens = 0;
  const keptMessages = [];

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
    service: 'OpenAI → NVIDIA NIM Proxy (Speed Optimized)'
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

// Chat completions (OTIMIZADO)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // 🔍 Salva para debug ANTES de qualquer processamento
    saveDebugEntry(req.body);

    const nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';
    const limitedMessages = limitMessagesByTokens(messages, 100000);

    const nimRequest = {
      model: nimModel,
      messages: limitedMessages,
      temperature: temperature ?? 1.0,
      max_tokens: max_tokens ?? 16384,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: !!stream
    };

    // ⚡ TIMEOUT EXTREMO para DeepSeek (pode demorar 5+ minutos!)
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json',
      timeout: 600000 // ⚠️ 10 MINUTOS (extremo!)
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
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));

            if (!SHOW_REASONING && data.choices?.[0]?.delta?.reasoning_content) {
              delete data.choices[0].delta.reasoning_content;
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
            content: choice.message?.content || ''
          },
          finish_reason: choice.finish_reason
        })),
        usage: response.data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);

    if (error.code === 'ECONNABORTED') {
      console.error('Timeout - modelo demorou mais de 10 minutos');
    }

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
  console.log(`⚡ Proxy OTIMIZADO rodando na porta ${PORT}`);
  console.log(`🌐 Health: http://localhost:${PORT}/health`);
  console.log(`🔍 Debug: http://localhost:${PORT}/debug`);
  console.log(`🚀 Modo: Velocidade Máxima`);

  // Self-ping a cada 10 minutos para o Render não dormir
  const RENDER_URL = process.env.RENDER_EXTERNAL_URL;
  if (RENDER_URL) {
    setInterval(() => {
      axios.get(`${RENDER_URL}/health`)
        .then(() => console.log(`🏓 Keep-alive ping OK`))
        .catch(err => console.warn(`⚠️ Keep-alive falhou: ${err.message}`));
    }, 10 * 60 * 1000);
    console.log(`🏓 Keep-alive ativo → ${RENDER_URL}/health`);
  }
});
