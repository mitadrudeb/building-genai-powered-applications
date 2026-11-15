# 🚀 Building GenAI-Powered Applications

A curated collection of practical, production-oriented **Generative AI & Multimodal applications** built by [Mitadru Deb](https://github.com/mitadrudeb).

Each project in this repository explores a foundational pillar of modern AI engineering: from multimodal vision-language models and retrieval-augmented generation (RAG) to agentic workflows and fine-tuning — built with open-source tools, clean architecture, and interactive user interfaces.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📂 Projects Showcase

| # | Project | Domain | Key Technologies | Status | Link |
|---|---------|--------|------------------|:------:|:----:|
| **01** | **AI Image Captioner** | Multimodal Vision AI & Web Scraping | Salesforce BLIP (~0.25B), Gradio, BeautifulSoup, PyTorch | ✅ Completed | [📁 Project Folder](./01-ai-image-captioner) <br> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mitadrudeb/building-genai-powered-applications/blob/main/01-ai-image-captioner/AI_Image_Captioner.ipynb) |
| **02** | **Conversational AI Chatbot** | Conversational Dialogue & Context Memory | Facebook BlenderBot (~0.4B), PyTorch, Gradio, Transformers | ✅ Completed | [📁 Project Folder](./02-conversational-ai-chatbot) <br> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mitadrudeb/building-genai-powered-applications/blob/main/02-conversational-ai-chatbot/Conversational_AI_Chatbot.ipynb) |
| **03** | *Audio-to-Insight Transcriber* | Speech AI & Summarization | OpenAI Whisper, PyTorch, Gradio | 🚧 Planned | *Coming Soon* |
| **04** | *Function-Calling AI Agent* | Agentic Systems & Tool Use | Tool Calling, API Integration, Pydantic | 🚧 Planned | *Coming Soon* |
| **05** | *Lightweight LLM Fine-Tuner* | Parameter-Efficient Tuning | LoRA / QLoRA, PEFT, TRL, Hugging Face Hub | 🚧 Planned | *Coming Soon* |
| **06** | *Multimodal Search & Embeddings* | Vector Search & Similarity | CLIP, Vector Embeddings, Cosine Distance | 🚧 Planned | *Coming Soon* |

---

## 🌟 Core Engineering Principles

All projects across this repository adhere to standard industry practices:

1. **Open-Source First:** Prioritizing open-weight models from Hugging Face Hub over expensive proprietary APIs.
2. **Accessible Hardware:** Designed to run comfortably on standard consumer CPUs, while scaling seamlessly to CUDA GPUs when available.
3. **One-Click Cloud Execution:** Every project includes a pre-configured Google Colab notebook badge for zero-friction testing.
4. **Interactive UIs:** Interactive Gradio frontends with shareable tunneling links rather than static scripts.
5. **Defensive Programming:** Thorough handling of network edge cases, tensor scalar types, and rate limits.

---

## 🛠️ Global Tech Stack

- **Frameworks & Backends:** PyTorch, Hugging Face Transformers, Hugging Face Hub
- **Vision & Multimodal:** Salesforce BLIP, PIL (Pillow), Torchvision
- **Data Ingestion & Scraping:** Requests, BeautifulSoup4, urllib
- **UI & Deployment:** Gradio, Google Colab
- **Version Control & Structure:** Git, GitHub Monorepo

---

## 🚀 Getting Started

Clone the repository to explore or run any project locally:

```bash
git clone https://github.com/mitadrudeb/building-genai-powered-applications.git
cd building-genai-powered-applications
```

Navigate to any specific project directory to review its standalone documentation and dependencies:

```bash
cd 01-ai-image-captioner
pip install -r requirements.txt
jupyter notebook AI_Image_Captioner.ipynb
```

---

## 👤 Author

**Mitadru Deb**  
- GitHub: [@mitadrudeb](https://github.com/mitadrudeb)

## 📄 License

This repository is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
