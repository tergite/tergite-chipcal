# Run a local vision model with Ollama

This page serves Qwen3.8 (27B, vision and thinking) on your own machine with [Ollama](https://ollama.com) and runs a
tergite-chipcal episode against it, through Ollama's OpenAI-compatible endpoint.

**Hardware.** The default `qwen3.8:27b` tag is about 18 GB (4-bit). A GPU with 24 GB or more holds it fully.
With less, Ollama moves layers to the CPU and runs much slower. An episode has up to 150 turns (2q1c), each a
thinking pass of up to 4096 tokens plus a short action pass, so expect hours per episode on one consumer GPU.

## 1. Install and serve

```bash
curl -fsSL https://ollama.com/install.sh | sh      # or see https://ollama.com/download
OLLAMA_CONTEXT_LENGTH=32768 ollama serve           # keep it running; listens on http://127.0.0.1:11434
```

Set the context length: a tergite-chipcal request is about 17k tokens (system prompt, observation, plot and the reasoning
memo of the two-pass client), and Ollama's default context is much shorter, which silently truncates the prompt.
If Ollama runs as a systemd service instead, set it there: `sudo systemctl edit ollama`, add
`Environment="OLLAMA_CONTEXT_LENGTH=32768"` under `[Service]`, then `sudo systemctl restart ollama`.

## 2. Pull the model

```bash
ollama pull qwen3.8:27b
ollama show qwen3.8:27b        # Capabilities should list vision and thinking
```

## 3. Run an episode

```bash
pip install -e ".[llm]"
python -m chipcal.run --tier 2q1c --seed 3000000 --out runs/ollama-demo --max-turns 10 \
    --base-url http://127.0.0.1:11434/v1 --model qwen3.8:27b --api-key ollama
```

`--max-turns 10` makes a short first test; drop it for a full episode. The run resumes from its checkpoint if
interrupted (run the same command again). `runs/ollama-demo/` then holds `transcript.json` (every turn, with the
model's reasoning), `run.json` (the score) and `vision/` (every plot the model was shown).

To evaluate on the frozen benchmark chips, one episode at a time against the single local server:

```bash
python scripts/evaluate.py --policy model --model qwen3.8:27b --base-url http://127.0.0.1:11434/v1 \
    --api-key ollama --tier 2q1c --n 10 --workers 1 --out results/ollama-qwen38
```

For a model without vision, add `--text-only` to either command: a text analysis of each measurement replaces the
plot.
