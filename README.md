# ProiectAI

Terminal quiz app powered by a small knowledge graph and a local LLM (llama.cpp) for question generation.

## Quick Start
1. `git pull` – make sure you are on the latest code.
2. `make setup` – creates `.venv` and installs the pinned tooling (`llama-cpp-python`, `huggingface_hub`, etc.).
3. `cp .env.example .env` – only if the chosen model repo requires a Hugging Face token, otherwise skip.
4. `make model` – downloads the GGUF declared in `model.lock` to `models/`.
5. `make run` – launches the interactive quiz.

The download step can also happen automatically the first time you run `script1.py`; the script checks `models/` and calls `scripts/get_model.py` if no `.gguf` is found.

## model.lock
`model.lock` is the single source of truth for which GGUF file the team should use. Update it (via PR) when switching models and, optionally, add the SHA-256 so everybody verifies the same binary. The downloader understands private/acceptance-gated repos via the optional `HUGGINGFACE_TOKEN`.

## Generated Questions
`script1.py` loads the local model through `llama_cpp` and asks it to produce one question per `(problem, property)` pair. If the model is unavailable for any reason, the app falls back to succinct in-code templates so the quiz still works offline.
