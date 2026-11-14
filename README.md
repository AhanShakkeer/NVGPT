# NVGPT

**NVGPT** means **Next Version Generative Pre-trained**. This repository starts the all-in-one AI platform with **AI Chat first**. Image generation, coding tools, document Q&A, and voice are intentionally left for later phases.

## Cloudflare Pages deployment

This is a static site with a Pages Function at `/api/chat`. It uses Cloudflare Workers AI; the model call happens on the server side, not in the browser.

1. In Cloudflare, open the Pages project connected to this repository.
2. In **Settings → Bindings**, add a **Workers AI** binding named exactly `AI`. The `wrangler.toml` also declares this binding for deployments that use Wrangler configuration.
3. Set the build output directory to `public`. Leave the build command empty; no build step is needed.
4. Save and redeploy. Cloudflare's AI availability and usage limits depend on your account and current plan.

The repository is connected to GitHub, so commits to `main` should trigger a new Pages deployment automatically. If this project was created with settings that differ from the steps above, check its build log/settings before changing them.

## Local development

Use Wrangler to run the Pages site and its function with the Workers AI binding. You will need to authenticate Wrangler with a Cloudflare account that has Workers AI available.

```sh
npx wrangler pages dev ./public
```

## Current scope

- Responsive NVGPT chat interface
- Recent chats in memory for the current browser session
- Server-side chat requests through a Pages Function
- Basic request validation and friendly error handling
- No login, database, payments, image generation, document uploads, or voice features yet

Do not put provider credentials or secrets in `public/index.html`. Future features should be added one at a time, with clear privacy and spending controls.
