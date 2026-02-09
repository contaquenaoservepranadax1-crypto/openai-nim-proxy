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
const NIM_API_KEY = process.env.NIM_API_KEY;

// Configurações de controle
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;

// Model mapping - ATUALIZADO com DeepSeek V3.1 base (funciona!)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',  // ⚡ Rápido
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',              // 💭 Emocional
  'gpt-4-turbo': 'deepseek-ai/deepseek-v3.1',                  // 🧠 DeepSeek V3.1 BASE (funciona!)
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',                       // 🧠 DeepSeek V3.1 BASE (principal)
  'gpt-4o-nemotron': 'nvidia/llama-3.1-nemotron-ultra-253b-v1', // ⚡ Backup rápido
  'claude-3-opus': 'qwen/qwen3-next-80b-a3b-thinking',         // 🤔 Teste
  'gemini-pro': 'nvidia/llama-3.1-nemotron-ultra-253b-v1'      // ⚡ Estável
};

// Estimativa de tokens (simplificada)
function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

// Limite adaptativo de histórico (AUMENTADO)
function limitMessagesByTokens(messages, maxTokens = 30000) {
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

    const nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';
    const limitedMessages = limitMessagesByTokens(messages, 30000); // ✅ 30k tokens = ~250-300 mensagens

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

    // STREAM MODE (SIMPLIFICADO - sem limpeza)
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
            
            // Remove reasoning se não quiser mostrar
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

    // NORMAL MODE (SIMPLIFICADO - sem limpeza)
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
    console.error('Proxy error:', error.message);
    
    // Log detalhado para debug
    if (error.code === 'ECONNABORTED') {
      console.error('Timeout - modelo demorou mais de 3 minutos');
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
  console.log(`🚀 Modo: Velocidade Máxima`);
});
