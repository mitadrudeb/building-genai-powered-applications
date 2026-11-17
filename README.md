# 🚀 Building GenAI-Powered Applications

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mitadrudeb01/)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-orange)
![Gradio](https://img.shields.io/badge/Gradio-UI-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A curated collection of practical, production-oriented **Generative AI, Large Language Model (LLM), and Multimodal applications** engineered by [Mitadru Deb](https://www.linkedin.com/in/mitadrudeb01/).

Each project in this monorepo explores a core pillar of modern AI engineering — from vision-language understanding and conversational dialogue systems with context memory, to retrieval-augmented generation (RAG) and agentic workflows — built with open-source models, clean architectures, and interactive web interfaces.

---

## 💼 Technical Competencies & Skills Demonstrated

| Competency Area | Skills & Techniques | Demonstrated In |
| :--- | :--- | :---: |
| **Large Language Models (LLMs)** | Conversational Memory (Sliding Window), Tokenizer Delimiters, Decoding Controls (Temperature, Top-P, Repetition Penalty, N-gram blocking) | **Project 02** |
| **Multimodal Vision-Language AI** | Vision Transformers (ViT), Cross-Attention Decoders, Web Scraping & Ingestion, Dynamic Batch Processing | **Project 01** |
| **Frontier API Integration** | OpenAI Chat Completions SDK (`openai>=1.0.0`), Message Schema Mapping, Secure Environment Configuration | **Project 02** |
| **Full-Stack AI Interfaces** | Gradio Web Apps, Interactive Parameter Sliders, Terminal CLI modes, Defensive Data Parsing | **Projects 01 & 02** |
| **Cloud & Deployment** | Google Colab 1-Click Reproducibility, PyTorch CUDA GPU / CPU auto-scaling, Git Monorepo architecture | **All Projects** |

---

## 📂 Projects Showcase

| # | Project | Domain | Key Technologies | Status | Link |
|---|---------|--------|------------------|:------:|:----:|
| **01** | **AI Image Captioner** | Multimodal Vision AI & Web Scraping | Salesforce BLIP (~0.25B), Gradio, BeautifulSoup, PyTorch | ✅ Completed | [📁 Project Folder](./01-ai-image-captioner) <br> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mitadrudeb/building-genai-powered-applications/blob/main/01-ai-image-captioner/AI_Image_Captioner.ipynb) |
| **02** | **Conversational AI Chatbot** | Conversational Dialogue & Context Memory | Facebook BlenderBot (~0.4B), PyTorch, Gradio, Transformers, OpenAI API | ✅ Completed | [📁 Project Folder](./02-conversational-ai-chatbot) <br> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mitadrudeb/building-genai-powered-applications/blob/main/02-conversational-ai-chatbot/Conversational_AI_Chatbot.ipynb) |
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
5. **Defensive Programming:** Thorough handling of network edge cases, tensor scalar types, and framework data formats.

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
# Example: Run Project 02 (Conversational AI Chatbot)
cd 02-conversational-ai-chatbot
pip install -r requirements.txt
python app.py
```

---

## 👤 Author

**Mitadru Deb**  
- LinkedIn: [linkedin.com/in/mitadrudeb01](https://www.linkedin.com/in/mitadrudeb01/)  
- GitHub: [@mitadrudeb](https://github.com/mitadrudeb)  
- Portfolio: [building-genai-powered-applications](https://github.com/mitadrudeb/building-genai-powered-applications)

## 📄 License

This repository is distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
