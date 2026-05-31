# SPIRE — Karnataka Government Schemes AI Assistant

**Parameter-Efficient Fine-Tuning of Large Language Models for Domain-Specific Question Answering**

SPIRE is a domain-specific AI assistant that answers citizen queries about Karnataka state government schemes. It uses a fine-tuned **Llama 3.2 3B Instruct** model with **QLoRA (4-bit quantization)** to deliver accurate, structured information about eligibility, benefits, documents, and application procedures for 30+ government schemes.

Built by **Santhosh Gowda M**, Research Intern at **SPIRE Lab**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Fine-Tuning](#fine-tuning)
- [Model](#model)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Schemes Covered](#schemes-covered)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview

Government scheme information in India is often scattered across dozens of portals and PDFs, making it difficult for citizens to find accurate answers. SPIRE addresses this by:

1. **Curating a high-quality dataset** — Crawling 12 government portals and processing 140+ PDFs to create 1,200+ Q&A pairs across 30+ schemes
2. **Fine-tuning a small LLM** — Using QLoRA on Llama 3.2 3B to produce structured, factually-grounded answers
3. **Serving via a local chat interface** — A dark-mode web UI that connects to Ollama for fully offline inference

The project demonstrates that dataset curation quality has a **greater impact on performance than model size**, and that 4-bit quantization enables practical fine-tuning on consumer-grade GPUs (~5 GB VRAM).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser (karnataka_assistant.html)      │
│  ┌──────────┐  ┌──────────────────────────────────────────┐ │
│  │ Sidebar  │  │  Chat Area                               │ │
│  │ Quick Qs │  │  ┌─────────────────────────────────────┐ │ │
│  │ Settings │  │  │  Streaming responses with rich       │ │ │
│  │          │  │  │  key:value cards, lists, notes       │ │ │
│  └──────────┘  │  └─────────────────────────────────────┘ │ │
│                │  ┌─────────────────────────────────────┐ │ │
│                │  │  Input: textarea + send              │ │ │
│                │  └─────────────────────────────────────┘ │ │
│                └──────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │ POST /api/chat (stream=true)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Ollama Server (localhost:11434)            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  llama.cpp inference on GGUF model                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │ Loads GGUF file
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Karnataka-Llama-3.2-3B (Q4_K_M GGUF)             │
│  Fine-tuned on 4,309 Q&A pairs over Karnataka schemes       │
└─────────────────────────────────────────────────────────────┘
```

### Data Pipeline

```
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ Source   │───▶│ Extraction & │───▶│ Curation & │───▶│ Formatting & │
│ Ingestion│    │ Cleaning     │    │ Q&A Gen    │    │ Fine-Tune    │
└──────────┘    └──────────────┘    └────────────┘    └──────────────┘
      │                │                  │                  │
      ▼                ▼                  ▼                  ▼
  12 portals      140+ PDFs         1,200+ Q&A         4,309 SFT
  crawled          parsed            pairs curated      records
```

---

## Features

- **AI-powered Q&A** about 30+ Karnataka government schemes
- **Structured responses** with color-coded Key:Value cards (category, department, benefit, eligibility, documents, mode of application)
- **Streaming output** — real-time token-by-token response rendering
- **Quick questions** organized by category in a sidebar (Women Empowerment, Youth & Employment, Health, Education, Agriculture, Social Welfare)
- **Model management** — switch between available Ollama models via settings
- **Configurable parameters** — temperature (0–2), max tokens (256–8192), context window
- **Dark theme UI** — optimized for readability with custom card components
- **Settings persistence** via `localStorage`
- **Fully offline** — no internet required once Ollama and the model are set up

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Base Model** | [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (Meta) |
| **Fine-tuning Library** | [Unsloth](https://github.com/unslothai/unsloth) — 2x faster fine-tuning |
| **Quantization** | 4-bit QLoRA (NF4 + double quantization) |
| **Inference Engine** | [llama.cpp](https://github.com/ggml-org/llama.cpp) via [Ollama](https://ollama.ai) |
| **Frontend** | Single-file HTML + CSS + JavaScript (vanilla) |
| **Training Hardware** | Google Colab — Tesla T4 (14.6 GB VRAM) |
| **ML Framework** | PyTorch, Transformers, TRL, Datasets, bitsandbytes |
| **Model Format** | GGUF — `q4_k_m` quantization |

---

## Dataset

### Source
- **12 Karnataka government portals** crawled for scheme information
- **140+ PDFs** processed (scheme guidelines, application forms, eligibility criteria)
- **85 forms** collected from district and state-level portals

### Construction Pipeline

```
1. Scraping ──▶ Automated extraction from govt websites
2. Cleaning ──▶ OCR cleanup, duplicate removal, eligibility normalization
3. Curation ──▶ Manual verification and correction by domain experts
4. Q&A Gen ──▶ Fields converted to natural-language question-answer pairs
5. Format ────▶ Llama-3.1 chat template with response-only loss masking
```

### Dataset Format

Each record follows an instruction-response format:

```json
{
  "instruction": "Who is eligible for Gruha Lakshmi Scheme?",
  "input": "",
  "output": "Gruha Lakshmi Scheme is a Karnataka government scheme. Category: Social Welfare. Managing Department: Department of Women and Child Development. Key Benefit: ₹2,000 per month financial assistance. Eligibility: Women heads of households from BPL families, Karnataka resident, age 18+. Required Documents: Aadhaar card, BPL certificate, bank account details. Mode of Application: Online via Seva Sindhu portal or offline at Grama One centers."
}
```

| Metric | Value |
|---|---|
| Government portals crawled | 12 |
| PDFs processed | 140+ |
| Final Q&A pairs | 1,200+ |
| SFT dataset records | 4,309 |
| Languages | English + Kannada |

---

## Fine-Tuning

### Configuration

| Parameter | Value |
|---|---|
| **Model** | `unsloth/Llama-3.2-3B-Instruct` |
| **Max sequence length** | 2048 |
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0 |
| **Target modules** | Q, K, V, O, Gate, Up, Down projections |
| **Batch size** | 2 |
| **Gradient accumulation** | 4 steps |
| **Effective batch size** | 8 |
| **Epochs** | 3 |
| **Total steps** | 1,617 |
| **Optimizer** | AdamW 8-bit |
| **Learning rate** | 2e-4 |
| **LR scheduler** | Cosine |
| **Warmup steps** | 50 |
| **Weight decay** | 0.01 |
| **Trainable params** | 24.3M (0.75% of 3.24B) |
| **Quantization** | 4-bit NF4 (QLoRA) |
| **Training time** | ~56 minutes (Tesla T4) |
| **Final training loss** | 0.0372 |

### Key Techniques

- **4-bit QLoRA**: NF4 quantization reduces VRAM to ~5 GB while preserving weight distribution fidelity. Double quantization further compresses scaling factors.
- **Response-only loss masking**: Loss is computed only on assistant response tokens — not on system prompts or user questions — reducing prompt memorization and improving generalization.
- **LoRA rank 16**: Achieves the best trade-off between adaptation quality and memory footprint. Higher ranks (32, 64) show diminishing returns.

### Loss Curve

Training started at ~0.15 loss and converged smoothly to **0.0372** by step 1,617 with no overfitting observed.

---

## Model

The fine-tuned model is exported in **GGUF q4_k_m** format for efficient CPU/GPU inference via llama.cpp and Ollama.

| Attribute | Value |
|---|---|---|
| **Base** | Llama 3.2 3B Instruct |
| **Fine-tuned name** | `Karnataka-Llama-3.2-3B` |
| **Format** | GGUF (q4_k_m) |
| **File size** | ~1.93 GB |
| **Inference temperature** | 0.2–0.5 (recommended) |
| **Context window** | up to 8192 tokens |
| **Ollama Modelfile** | Includes Llama 3.1 chat template, temperature = 1.5, min_p = 0.1 |
| **Download** | [Google Drive](https://drive.google.com/file/d/1yKdz4H2qL3ViIakIXZ17m1C_49ls7xj2/view?usp=drive_link)

### Response Format

The model is trained to produce structured answers in this format:

```text
The Shakti Scheme provides free bus travel for women in Karnataka.
Category: Social Welfare / Women Transport Empowerment
Managing Department: Department of Transport, Government of Karnataka
Key Benefit: Free, unlimited travel on all KSRTC/NEKRTC/NWKRTC buses
Eligibility: All women residents of Karnataka with valid Karnataka photo ID
Required Documents: Aadhaar card, Karnataka address proof, passport photo
Mode of Application: Offline — apply at KSRTC depot or Grama One center
Level: State

> Note: Smart card issuance may take up to 90 days. Travel is free with any valid Karnataka photo ID in the interim.
```

---

## Quick Start

### Prerequisites

- [Ollama](https://ollama.ai) installed and running
- ~2 GB free disk space for the GGUF model

### Setup

#### Option 1: Use the pre-trained model (GGUF)

1. **Download the GGUF model and Modelfile** (from the `Karnataka-Llama-3.2-3B-GGUF_gguf/` directory)

2. **Create the Ollama model**

   ```bash
   cd Karnataka-Llama-3.2-3B-GGUF_gguf
   ollama create karnataka-llama3 -f Modelfile
   ```

3. **Launch the web UI**

   Open `karnataka_assistant.html` in any browser. The UI will auto-connect to `http://localhost:11434`.

#### Option 2: Fine-tune from scratch (Google Colab)

1. Open `Llama_3_2_3B_Instruct_FIneTuning.ipynb` in Google Colab
2. Ensure you have your `karnataka_schemes_sft_dataset.jsonl` in the Colab environment
3. Run all cells (Tesla T4 takes ~56 minutes)
4. The notebook will output:
   - LoRA adapters: `Karnataka-Llama-3.2-3B-LoRA/`
   - Merged fp16 model: `Karnataka-Llama-3.2-3B-Merged/`
   - GGUF file: `Karnataka-Llama-3.2-3B-GGUF/`
5. Copy the GGUF to your local machine and follow Option 1 steps 2–3

### Running

1. Start Ollama:
   ```bash
   ollama serve
   ```

2. Open `karnataka_assistant.html` in your browser.

3. (Optional) Click the settings gear icon to:
   - Change the Ollama URL
   - Select a different model
   - Adjust temperature (0.2–0.5 recommended for factual answers)
   - Adjust max tokens (4096 recommended)

---

## Usage

### Ask questions naturally

Type any question about Karnataka government schemes, for example:

- *"Who can apply for Gruha Lakshmi?"*
- *"What documents are needed for Yuva Nidhi?"*
- *"How to apply for Shakti Scheme?"*
- *"Krushi Bhagya Scheme subsidy details?"*
- *"Anna Bhagya Scheme benefits?"*

### Use quick questions

The sidebar contains categorized quick-question buttons for:
- **Women Empowerment** — Gruha Lakshmi, Shakti, Udyogini, Bhagyalakshmi
- **Youth & Employment** — Yuva Nidhi, Self Employment
- **Health & Insurance** — Arogya Karnataka, Vajpayee Arogya Shree, Santwana Harish
- **Education & Scholarships** — Pre-matric SC/ST, Arivu, Prabhuddha Overseas, Vidyasiri
- **Agriculture & Farmers** — Krushi Bhagya, Ganga Kalyana, Krushi Aranya
- **Social Welfare** — Anna Bhagya, SC Marriage Assistance, Gruha Jyoti

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Shift+Enter` | New line in input |

---

## Schemes Covered

| Category | Schemes |
|---|---|
| **Women Empowerment** | Gruha Lakshmi, Shakti (free bus travel), Udyogini, Bhagyalakshmi |
| **Youth & Employment** | Yuva Nidhi, Karnataka Self Employment |
| **Health & Insurance** | Arogya Karnataka, Vajpayee Arogya Shree, Mukhyamantri Santwana Harish |
| **Education & Scholarships** | Pre-matric SC/ST, Arivu Education Loan, Prabhuddha Overseas, Vidyasiri |
| **Agriculture & Farmers** | Krushi Bhagya, Ganga Kalyana, Krushi Aranya Protsaha Yojane |
| **Social Welfare** | Anna Bhagya (free rice), SC Marriage Assistance, Gruha Jyoti (free electricity) |
| **General** | Five Guarantees, BPL schemes, Minority schemes, Farmer subsidies |

---

## Results

### Before vs After Fine-Tuning

| Aspect | Before Fine-Tuning | After Fine-Tuning |
|---|---|---|
| Response structure | Generic, incomplete | Structured key:value cards |
| Eligibility details | Missing or vague | Precise, categorized |
| Hallucination rate | High | Low |
| Response consistency | Poor | High |
| Scheme-specific accuracy | Low | 90%+ factual |

### Key Findings

1. **Dataset quality > Model size** — Curation impact exceeded model size in determining performance
2. **QLoRA + 4-bit NF4** enabled stable low-resource fine-tuning at ~5 GB VRAM
3. **Response-only loss masking** significantly improved generalization

### Comparison with Other Models

| Model | Strength | VRAM | Observation |
|---|---|---|---|
| **Gemma-3** | Strong reasoning | ~5 GB | Best balance of accuracy and efficiency |
| **Qwen-2.5** | Multilingual | ~6 GB | Better conversational responses |
| **Llama-3.2** | Instruction tuning | ~8 GB | High-quality outputs, higher compute |
| **OpenHathi-7B** | Hindi-centric | ~5 GB | Weak generalization on scheme queries |

---

## Project Structure

```
SPIRE_Project/
├── karnataka_assistant.html              # Standalone web chat UI (1926 lines)
├── Llama_3_2_3B_Instruct_FIneTuning.ipynb  # Full fine-tuning pipeline (Colab)
├── SPIRE_Presentation.pptx               # Research presentation (12 slides)
├── model.zip                             # Backup of GGUF model + Modelfile
└── Karnataka-Llama-3.2-3B-GGUF_gguf/     # Ollama-ready model directory
    ├── llama-3.2-3b-instruct.Q4_K_M.gguf # Quantized model (~1.93 GB)
    └── Modelfile                         # Ollama model configuration
```

---

## Challenges & Roadmap

### Challenges

- **Noisy PDFs** — OCR inconsistencies across government portals
- **Website formatting differences** — Limited structured datasets across sources
- **Hallucination control** — Low-resource domain setting required careful loss masking

### Future Roadmap

- Expand to all Indian state schemes
- Add Kannada language support throughout the UI
- Implement multi-turn conversation memory
- Deploy as a government kiosk application
- Add RAG (Retrieval-Augmented Generation) for dynamic document lookup

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{spire-karnataka-assistant,
  author = {Santhosh Gowda M},
  title = {SPIRE: Parameter-Efficient Fine-Tuning of LLMs for Karnataka Government Scheme Question Answering},
  year = {2025},
  publisher = {SPIRE Lab}
}
```

---

## License

This project is for research and educational purposes. The fine-tuned model weights are derived from Meta's Llama 3.2 and are subject to its original license terms. Government scheme data is sourced from publicly available Karnataka government portals.

---

*Built with Unsloth, Ollama, and ❤️ at SPIRE Lab.*
