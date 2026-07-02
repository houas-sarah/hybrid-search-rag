# Deploying to Hugging Face Spaces

This app loads PyTorch plus two models, so it needs more RAM than most free
tiers give you. Hugging Face Spaces (Docker, free CPU) has 16 GB and is built for
this, so that's what these notes cover. It runs on the **Groq** backend (Ollama
can't run on a free host).

## 1. Create the Space

Go to <https://huggingface.co/new-space>:

- **Owner / name**: `hybrid-search-rag`
- **SDK**: **Docker** → *Blank*
- **Hardware**: CPU basic (free)

## 2. Push the code

The Space is its own git repo. Add it as a remote and push:

```bash
git remote add space https://huggingface.co/spaces/sarahouas/hybrid-search-rag
git push space main
```

When git asks for credentials, use your Hugging Face username and a **write
token** (create one at <https://huggingface.co/settings/tokens>).

## 3. Tell HF it's a Docker app

Pushing overwrote the Space's `README.md` with this repo's one, which HF doesn't
recognize as a Space config. Open the Space → **Files** → edit `README.md` and
paste this block at the very top, then commit:

```
---
title: Hybrid Search RAG
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---
```

(The Dockerfile already runs uvicorn on `0.0.0.0:8000`, which matches `app_port`.)

## 4. Set the environment

Space → **Settings** → *Variables and secrets*:

| Kind | Name | Value |
|------|------|-------|
| Secret   | `GROQ_API_KEY`      | your Groq key |
| Variable | `LLM_PROVIDER`      | `groq` |
| Variable | `INGEST_ON_STARTUP` | `true` |

`INGEST_ON_STARTUP=true` makes the app build the index on boot, since the Space's
disk is wiped on each restart/rebuild.

## 5. Wait for the build

The first build takes a few minutes (installing torch + deps, then pulling the
models on first boot). When it's green, the app is live at:

```
https://sarahouas-hybrid-search-rag.hf.space
```

Put that URL in the GitHub repo's **About → Website**.

## Notes

- A free Space sleeps after a while of inactivity; the next visit cold-starts and
  reloads the models (~30 s).
- Anyone using the live demo hits **your** Groq key server-side, so it counts
  against your free quota. Fine for a low-traffic portfolio demo.
- To redeploy after changes: `git push space main`.
