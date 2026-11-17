# 🤖 Conversational AI Chatbot with Multi-Turn Memory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mitadrudeb/building-genai-powered-applications/blob/main/02-conversational-ai-chatbot/Conversational_AI_Chatbot.ipynb)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-BlenderBot_400M-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

An open-source, empathetic conversational dialogue agent powered by **`facebook/blenderbot-400M-distill`**. Features dynamic **multi-turn sliding-window conversational memory**, tunable decoding parameters (temperature, nucleus sampling, repetition penalty), and dual interfaces: an interactive **Gradio Web UI** and a lightning-fast **Terminal CLI mode**.

Part of the [Building GenAI-Powered Applications](https://github.com/mitadrudeb/building-genai-powered-applications) portfolio series by [Mitadru Deb](https://github.com/mitadrudeb).

---

## 📌 Project Highlights

- **🧠 Multi-Turn Conversational Memory:** Implements a sliding-window memory buffer that tracks past dialogue turns, resolves pronouns across turns, and prevents context token overflows.
- **⚡ Lightweight & CPU-Friendly:** Powered by `facebook/blenderbot-400M-distill` (~400M parameters, ~730 MB). Generates responses in under **1 second on consumer CPUs** with automatic scaling to NVIDIA CUDA GPUs.
- **🔒 100% Open-Source & Zero API Keys:** Runs completely offline and locally without paying for proprietary OpenAI/Anthropic APIs or managing rate limits.
- **🎛️ Dynamic Generation Controls:** Real-time sliders in Gradio for **Temperature**, **Top-P (Nucleus Sampling)**, **Repetition Penalty**, **Max Tokens**, and **Context Window Depth**.
- **💻 Dual Interaction Modes:** Run it as a rich web app with `python app.py` or as a terminal chat assistant with `python app.py --cli`.
- **🌐 Frontier Model Bonus:** Includes a modular OpenAI Chat Completions integration block to demonstrate how the same conversational memory translates to proprietary models (e.g. GPT-4o-mini).
- **🚀 1-Click Cloud Execution:** Pre-configured Google Colab notebook for instant execution with free GPU support.

---

## 🏛️ Architecture & How It Works

Large Language Models and conversational agents are fundamentally **stateless** — they retain zero knowledge between inference calls. Statefulness is an engineering abstraction built by passing dialogue history into the prompt.

```
┌────────────────────────────────────────────────────────┐
│                   User Dialogue Turn                   │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│             Sliding-Window Memory Buffer               │
│  [Turn t-2]  ──>  [Turn t-1]  ──>  [Current Message]   │
│  (Truncated to last N turns to respect 512 token limit)│
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│        Native BlenderBot Delimiter Formatting          │
│           "Utterance 1  Utterance 2  Utterance 3"      │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│      Hugging Face AutoModelForSeq2SeqLM (400M)         │
│          Sampling (Temp, Top-P, Rep-Penalty)           │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│            Gradio Web UI / Terminal CLI Output         │
└────────────────────────────────────────────────────────┘
```

### Why the Native Delimiter Matters
BlenderBot was pre-trained using **double spaces (`"  "`)** as the separator between conversational turns. Formatting prompts with artificial labels like `User:` or `Bot:` causes out-of-distribution confusion in distilled models, leading to conversational resets. Using the native delimiter preserves natural, context-aware responses across turns.

---

## ⚙️ Generation Parameter Cheatsheet

| Parameter | What it Does | Typical Range | Recommended |
| :--- | :--- | :---: | :---: |
| **`temperature`** | Controls randomness. Lower values make outputs deterministic and focused; higher values increase creativity and vocabulary diversity. | `0.1 – 1.5` | `0.7` |
| **`top_p`** | Nucleus sampling: restricts candidate tokens to the smallest set whose cumulative probability reaches $p$. | `0.1 – 1.0` | `0.85` |
| **`repetition_penalty`** | Penalizes tokens that have already appeared in the prompt or output, preventing repetitive loops. | `1.0 – 2.0` | `1.3` |
| **`no_repeat_ngram_size`** | Hard block forbidding any $n$-gram of length 3 from repeating in the generated sequence. | `2 – 4` | `3` |
| **`history_turns`** | Number of previous dialogue turns retained in the prompt's sliding window. | `2 – 12` | `6` |
| **`max_new_tokens`** | Upper limit on the number of generated tokens per bot turn. | `20 – 150` | `60` |

---

## 📁 Directory Structure

```text
02-conversational-ai-chatbot/
├── app.py                         # Standalone script (Gradio Web UI + CLI mode + OpenAI bonus)
├── Conversational_AI_Chatbot.ipynb # Interactive Jupyter Notebook (Google Colab ready)
├── requirements.txt               # Lightweight project dependencies
└── README.md                      # Project documentation and guide
```

---

## 🚀 Quickstart & Setup

### 1. Local Setup

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/mitadrudeb/building-genai-powered-applications.git
cd building-genai-powered-applications/02-conversational-ai-chatbot
```

Create a virtual environment (optional but recommended) and install dependencies:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch the Gradio Web App

```bash
python app.py
```
Open `http://localhost:7860` in your browser to start chatting with the interactive UI.

### 3. Run in Terminal CLI Mode

If you prefer chatting directly in your terminal:

```bash
python app.py --cli
```

Commands in CLI mode:
- `clear`: Resets conversational memory history.
- `exit` or `quit`: Exits the chat session.

### 4. Run in Google Colab (Zero Installation)

Click the badge at the top or open:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mitadrudeb/building-genai-powered-applications/blob/main/02-conversational-ai-chatbot/Conversational_AI_Chatbot.ipynb)

Run all cells step-by-step to experiment with code, visualize conversational memory, and launch the shareable Gradio link.

---

## 🌐 Bonus: Frontier Model Integration (OpenAI API)

To showcase knowledge of both open-source models and frontier commercial APIs, a modular integration section is included in both `app.py` and `Conversational_AI_Chatbot.ipynb`.

Anyone with their own **OpenAI API key** can test this section with models like **GPT-4o-mini** or **GPT-4o**:

1. Install the official OpenAI SDK:
   ```bash
   pip install openai
   ```
2. Set your environment variable:
   ```bash
   # Linux / macOS:
   export OPENAI_API_KEY="your-api-key-here"

   # Windows PowerShell:
   $env:OPENAI_API_KEY="your-api-key-here"
   ```
3. Uncomment the `generate_openai_response` block in `app.py` or run Section 8 in the Jupyter Notebook.

---

## 🧪 Experiments, Failure Modes & Key Observations

During the implementation and testing of this conversational agent, I conducted targeted experiments to analyze model behavior, memory retention, and decoding dynamics:

1. **Pronoun Resolution & Multi-Turn Coherence:**
   - *Test:* Asked *"I am planning a trip to Kyoto next spring."* followed by *"What are the best temples to see there?"*
   - *Observation:* With sliding-window memory active, the model successfully resolved the pronoun *"there"* to Kyoto without requiring explicit repetition, recommending popular historical sites (such as Kinkaku-ji and Fushimi Inari).
2. **Decoding Parameter Sensitivity (Temperature & Repetition Penalty):**
   - *Low Temperature (0.1 – 0.3):* The model produces deterministic, conservative responses, but frequently repeats boilerplate greetings or stock phrases.
   - *Balanced Temperature (0.6 – 0.8) + Repetition Penalty (1.2 – 1.3):* Delivers the most natural, varied conversational flow while preventing token looping.
   - *High Temperature (>1.2):* Creative but prone to hallucination and topic drift.
3. **Sliding-Window Memory Truncation:**
   - Setting `history_turns=1` caused the model to lose context immediately when asked reference questions (e.g., *"What was the first topic we discussed?"*).
   - Setting `history_turns=6` retained sufficient context for natural back-and-forth chit-chat while remaining well below BlenderBot's 512-token sequence limit.
4. **Delimiter Sensitivity (BlenderBot-Specific):**
   - In early testing, injecting artificial role markers (`User:` / `Bot:`) caused out-of-distribution confusion in BlenderBot's BPE tokenizer, leading to conversational resets.
   - Aligning with BlenderBot's native double-space (`"  "`) delimiter immediately resolved this issue and restored multi-turn topic coherence.

---

## 🏆 Engineering Takeaways & Lessons Learned

1. **Statefulness is an Application-Level Abstraction:**
   Transformer models are inherently stateless function approximators. Conversational continuity requires deliberate engineering: capturing, formatting, and injecting past dialogue into the input representation before each forward pass.
2. **Sliding-Window Context Bounding:**
   Unbounded conversation history leads to quadratic self-attention costs and context window truncation. Implementing an explicit sliding window keeps computation bounded, latency predictable, and token counts within the model's budget.
3. **Training Distribution Alignment:**
   Prompt structure cannot be arbitrary. Distilled dialogue models like BlenderBot are sensitive to formatting; adhering strictly to their pre-training delimiter (`"  "`) was essential to prevent persona degradation.
4. **Defensive API & UI Design:**
   Frameworks like Gradio 5/6 pass structured multimodal content blocks (`list` / `dict`) rather than simple strings across turns. Implementing defensive text extraction (`extract_text`) prevents production runtime crashes.

---

## 👤 Author

**Mitadru Deb**  
- GitHub: [@mitadrudeb](https://github.com/mitadrudeb)  
- Portfolio: [building-genai-powered-applications](https://github.com/mitadrudeb/building-genai-powered-applications)

## 📄 License

Distributed under the **MIT License**. See [LICENSE](../LICENSE) for details.
