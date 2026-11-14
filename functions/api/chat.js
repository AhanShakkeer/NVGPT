const MAX_MESSAGES = 20;
const MAX_MESSAGE_LENGTH = 8000;
const MODEL = '@cf/meta/llama-3.1-8b-instruct';

export async function onRequestPost({ request, env }) {
  if (!env.AI) {
    return json({ error: 'AI is not configured yet. In Cloudflare, add an AI binding named AI to this Pages project, then retry.' }, 503);
  }

  let payload;
  try {
    payload = await request.json();
  } catch {
    return json({ error: 'Please send a valid chat message.' }, 400);
  }

  if (!Array.isArray(payload.messages) || payload.messages.length === 0 || payload.messages.length > MAX_MESSAGES) {
    return json({ error: 'Send between 1 and 20 messages.' }, 400);
  }

  const messages = [];
  for (const message of payload.messages) {
    if (!message || !['user', 'assistant'].includes(message.role) || typeof message.content !== 'string') {
      return json({ error: 'The chat contains an invalid message.' }, 400);
    }
    const content = message.content.trim();
    if (!content || content.length > MAX_MESSAGE_LENGTH) {
      return json({ error: 'Each message must be between 1 and 8,000 characters.' }, 400);
    }
    messages.push({ role: message.role, content });
  }

  if (messages[messages.length - 1].role !== 'user') {
    return json({ error: 'The latest message must be from you.' }, 400);
  }

  try {
    const result = await env.AI.run(MODEL, {
      messages: [
        {
          role: 'system',
          content: 'You are NVGPT, a friendly, thoughtful AI assistant. NVGPT stands for Next Version Generative Pre-trained. Answer clearly and helpfully. Be honest about uncertainty. For coding questions, explain solutions carefully. Do not claim to have capabilities you do not have.'
        },
        ...messages
      ],
      max_tokens: 1200,
      temperature: 0.7
    });

    const reply = result?.response;
    if (typeof reply !== 'string' || !reply.trim()) {
      return json({ error: 'The AI did not return a reply. Please try again.' }, 502);
    }
    return json({ reply: reply.trim() });
  } catch (error) {
    console.error('NVGPT AI request failed:', error);
    return json({ error: 'NVGPT could not reach its AI model right now. Please try again in a moment.' }, 502);
  }
}

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store'
    }
  });
}
