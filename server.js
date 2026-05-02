// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '100mb' }));

app.use((req, res, next) => {
  req.socket.setKeepAlive(true, 15000);
  req.socket.setTimeout(0);
  next();
});

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NVIDIA_SECOND_API_KEY;

// DESATIVADO: thinking mode causa output de lixo/exclamacoes no JanitorAI
const ENABLE_THINKING_MODE = false;

const MODEL_MAPPING = {
  'gpt-3.5-turbo':   'moonshotai/kimi-k2.5',
  'gpt-4':           'deepseek-ai/deepseek-v3-0324',
  'gpt-4-turbo':     'moonshotai/kimi-k2.6',
  'gpt-4o':          'deepseek-ai/deepseek-v4-pro',
  'gpt-4o-nemotron': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus':   'z-ai/glm4.7',
  'gemini-pro':      'minimaxai/minimax-m2.5'
};

// ============================================================
// Debug store
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
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

app.get('/debug', (req, res) => {
  if (debugStore.length === 0) {
    return res.send(`<html><body style="font-family:monospace;padding:20px;background:#111;color:#0f0">
      <h2>Debug - Nenhum request recebido ainda</h2>
      <p>Faca uma mensagem no JanitorAI e recarregue esta pagina.</p>
    </body></html>`);
  }
  const entryIndex = Math.min(parseInt(req.query.entry || '0'), debugStore.length - 1);
  const entry = debugStore[entryIndex];
  const messagesHTML = entry.messages.map(m => `
    <div style="border:1px solid #333;margin:8px 0;padding:12px;border-radius:6px;background:#1a1a1a">
      <div style="margin-bottom:8px">
        <span style="background:${m.role==='system'?'#4a3000':m.role==='user'?'#003a4a':'#1a3a00'};padding:2px 8px;border-radius:4px;font-size:12px">[${m.index}] ${m.role.toUpperCase()}</span>
        <span style="color:#888;font-size:12px;margin-left:10px">${m.char_length} chars · ~${m.estimated_tokens} tokens</span>
      </div>
      <pre style="white-space:pre-wrap;word-break:break-word;color:#ccc;font-size:13px;margin:0">${escapeHtml(m.content_preview)}</pre>
    </div>`).join('');
  const allEntriesNav = debugStore.map((e, i) => `
    <a href="/debug?entry=${i}" style="color:${i===entryIndex?'#0f0':'#666'};margin-right:15px;text-decoration:none;font-size:12px">
      ${i===entryIndex?'> ':''}[${i}] ${e.timestamp} - ${e.total_messages} msgs
    </a>`).join('<br>');
  res.send(`<html><head><title>Proxy Debug</title><meta charset="utf-8">
    <style>body{font-family:monospace;padding:20px;background:#111;color:#eee}h2{color:#0f0}
    .stat{display:inline-block;background:#222;padding:6px 14px;border-radius:6px;margin:4px;font-size:13px}
    .stat span{color:#0f0;font-weight:bold}.nav{background:#1a1a1a;padding:12px;border-radius:6px;margin-bottom:20px;font-size:12px;line-height:2}
    </style></head><body>
    <h2>Proxy Debug</h2>
    <div class="nav"><b style="color:#888">Historico:</b><br>${allEntriesNav}</div>
    <div style="margin-bottom:16px">
      <div class="stat">Modelo pedido: <span>${entry.model_requested}</span></div>
      <div class="stat">Mapeado: <span>${entry.model_mapped}</span></div>
      <div class="stat">Total msgs: <span>${entry.total_messages}</span></div>
      <div class="stat">Tokens est.: <span>${entry.estimated_tokens.toLocaleString()}</span></div>
      <div class="stat">Temperature: <span>${entry.temperature??'default'}</span></div>
      <div class="stat">Max tokens: <span>${entry.max_tokens??'default'}</span></div>
      <div class="stat">Stream: <span>${entry.stream?'sim':'nao'}</span></div>
    </div>
    <h3 style="color:#0af">Mensagens (${entry.total_messages} total)</h3>
    ${messagesHTML}<br>
    <button onclick="location.reload()" style="background:#0f0;color:#000;border:none;padding:10px 20px;border-radius:6px;cursor:pointer;font-weight:bold;margin-right:10px">Atualizar</button>
    <a href="/debug/raw" style="background:#333;color:#eee;padding:10px 20px;border-radius:6px;text-decoration:none">Ver JSON bruto</a>
    </body></html>`);
});

app.get('/debug/raw', (req, res) => {
  if (debugStore.length === 0) return res.json({ message: 'Nenhum request recebido ainda.' });
  res.json(debugStore[0]);
});

// ============================================================
// Helpers
// ============================================================
function limitMessagesByTokens(messages, maxTokens = 100000) {
  if (!messages || messages.length === 0) return messages;
  let totalTokens = 0;
  const keptMessages = [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const tokens = estimateTokens(JSON.stringify(messages[i]));
    if (totalTokens + tokens <= maxTokens) { keptMessages.unshift(messages[i]); totalTokens += tokens; }
    else break;
  }
  return keptMessages;
}

// Remove campos de raciocinio interno do delta (streaming)
function sanitizeDelta(delta) {
  if (!delta) return delta;
  // Remove qualquer campo de reasoning/thinking para nao vazar ao cliente
  delete delta.reasoning_content;
  delete delta.thinking;
  delete delta.reasoning;
  return delta;
}

// Remove campos de raciocinio interno da mensagem (modo normal)
function sanitizeMessage(message) {
  if (!message) return message;
  delete message.reasoning_content;
  delete message.thinking;
  delete message.reasoning;
  return message;
}

// ============================================================
// Routes
// ============================================================
app.get('/health', (_, res) => res.json({ status: 'ok' }));

app.get('/v1/models', (_, res) => {
  res.json({ object: 'list', data: Object.keys(MODEL_MAPPING).map(m => ({
    id: m, object: 'model', created: Date.now(), owned_by: 'nvidia-nim-proxy'
  }))});
});

// ============================================================
// Chat completions
// ============================================================
app.post('/v1/chat/completions', async (req, res) => {
  const { model, messages, temperature, max_tokens, stream } = req.body;
  saveDebugEntry(req.body);

  const nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';
  const limitedMessages = limitMessagesByTokens(messages, 100000);

  const nimRequest = {
    model: nimModel,
    messages: limitedMessages,
    temperature: temperature ?? 1.0,
    max_tokens: max_tokens ?? 16384,
    stream: true // sempre stream para manter conexao ativa no Render
    // thinking mode DESATIVADO — causava output de lixo no JanitorAI
  };

  // Adiciona thinking mode apenas se explicitamente habilitado
  if (ENABLE_THINKING_MODE) {
    nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
  }

  try {
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: 'stream',
      timeout: 600000
    });

    // ---- MODO STREAM ----
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      let buffer = '';
      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith(':')) continue; // ignora comentarios SSE
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) { res.write('data: [DONE]\n\n'); continue; }
          try {
            const data = JSON.parse(line.slice(6));

            // Remove reasoning/thinking do delta antes de repassar
            if (data.choices?.[0]?.delta) {
              data.choices[0].delta = sanitizeDelta(data.choices[0].delta);
            }

            // So repassa se houver conteudo real (evita chunks vazios de thinking)
            const hasContent = data.choices?.[0]?.delta?.content !== undefined
              || data.choices?.[0]?.finish_reason;
            if (hasContent) {
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            }
          } catch {}
        }
      });
      response.data.on('end', () => { if (!res.writableEnded) res.end(); });
      response.data.on('error', err => {
        console.error('Stream error:', err.message);
        if (!res.writableEnded) res.end();
      });

    // ---- MODO NORMAL ----
    } else {
      let fullContent = '';
      let finishReason = 'stop';
      let usageData = null;
      let buffer = '';

      response.data.on('data', chunk => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (line.startsWith(':')) continue;
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            // Acumula apenas content, ignora reasoning_content
            if (data.choices?.[0]?.delta?.content) {
              fullContent += data.choices[0].delta.content;
            }
            if (data.choices?.[0]?.finish_reason) finishReason = data.choices[0].finish_reason;
            if (data.usage) usageData = data.usage;
          } catch {}
        }
      });
      response.data.on('end', () => {
        res.json({
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{ index: 0, message: { role: 'assistant', content: fullContent }, finish_reason: finishReason }],
          usage: usageData ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
        });
      });
      response.data.on('error', err => {
        console.error('Error (non-stream):', err.message);
        if (!res.headersSent) res.status(500).json({ error: { message: err.message } });
      });
    }

  } catch (error) {
    console.error('Proxy error:', error.response?.data || error.message);
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: { message: error.message || 'Internal server error', type: 'proxy_error', code: error.response?.status || 500 }
      });
    } else if (!res.writableEnded) {
      res.end();
    }
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, code: 404 } });
});

const server = app.listen(PORT, () => {
  console.log(`Proxy rodando na porta ${PORT}`);
  const RENDER_URL = process.env.RENDER_EXTERNAL_URL;
  if (RENDER_URL) {
    setInterval(() => {
      axios.get(`${RENDER_URL}/health`)
        .then(() => console.log('Keep-alive OK'))
        .catch(err => console.warn(`Keep-alive falhou: ${err.message}`));
    }, 10 * 60 * 1000);
  }
});

server.setTimeout(0);
server.keepAliveTimeout = 620000;
server.headersTimeout = 630000;
