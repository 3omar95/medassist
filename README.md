# 🩺 MedAssist: Evidence-Grounded Medical Q&A with QLoRA + LangChain RAG

*MedAssist* is a domain-tuned LLM that combines *Supervised Fine-Tuning (SFT)* using *QLoRA* with *Retrieval-Augmented Generation (RAG)* through LangChain to provide accurate, evidence-grounded answers to medical questions.  
It leverages *Mistral-7B-Instruct* as the base model and integrates *LangChain* for retrieval, conversational memory, and orchestration, with a *Gradio UI* for interactive exploration.

> ⚠ *Disclaimer:* MedAssist is a research and demonstration project only. It does not provide medical advice.

---

## 🚀 Project Overview

- Fine-tuned a large instruction model on *PubMedQA* and curated clinical guideline data.
- Implemented *QLoRA (Quantized Low-Rank Adaptation)* for efficient fine-tuning on limited GPUs.
- Built a *RAG pipeline* using LangChain with *Chroma vector store* and *MiniLM embeddings*.
- Added *conversation memory* to maintain context across user queries.
- Developed a *Gradio web interface* to compare *base vs fine-tuned* model outputs.

---

## 🧠 Technical Stack

| Layer | Component | Technology |
|-------|------------|-------------|
| *Base Model* | mistralai/Mistral-7B-Instruct-v0.3 | Hugging Face Transformers |
| *Fine-Tuning* | QLoRA (4-bit, LoRA rank = 8) | bitsandbytes, peft, trl.SFTTrainer |
| *Dataset* | PubMedQA + custom prompts | Hugging Face Datasets |
| *Retrieval* | RAG via LangChain + Chroma / FAISS | langchain, sentence-transformers |
| *Reranker* | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | cross-encoder |
| *Memory* | Conversational buffer memory | langchain.memory |
| *Interface* | Gradio (model-switching UI) | gradio |
| *Deployment* | Docker / Hugging Face Spaces | Dockerfile, requirements.txt |

---

## 🧩 Pipeline Summary

### 1️⃣ Data Preparation
- Downloaded and pre-processed *PubMedQA* dataset.
- Generated multiple *instruction-style prompts* and normalized answers.
- Cleaned and truncated long abstracts for efficient tokenization.

### 2️⃣ Fine-Tuning (QLoRA)
- Loaded base model in *4-bit quantized mode* with BitsAndBytesConfig.
- Applied *LoRA adapters* to projection layers (q_proj, k_proj, v_proj, o_proj).
- Trained using SFTTrainer (TRL) with:
  - packing=True
  - group_by_length=True
  - cosine learning rate scheduler  
- Completed training within *< 11 hours* using capped steps and gradient accumulation.

### 3️⃣ RAG Integration
- Indexed medical guideline PDFs and PubMed abstracts using *Chroma + MiniLM embeddings*.
- Implemented *cross-encoder reranker* to improve retrieval relevance.
- Connected retriever → LLM → guardrails pipeline through LangChain.

### 4️⃣ Conversational Memory
- Used ConversationBufferMemory to maintain recent context for multi-turn coherence.

### 5️⃣ Interface (Gradio)
- Interactive web UI with:
  - Input field for questions
  - Option to toggle *base* / *fine-tuned* model
  - Display of *citations* and *source snippets*

### 6️⃣ Evaluation & Validation
- Compared *fine-tuned* vs *base* model outputs on held-out PubMedQA samples.
- Manual and automated evaluation metrics:

---

## 🧩 Skills Demonstrated

| Area | Evidence |
|------|-----------|
| *LLM Fine-Tuning* | Implemented 4-bit QLoRA with PEFT and TRL SFTTrainer |
| *Prompt & Data Engineering* | Created multi-prompt SFT dataset for biomedical reasoning |
| *LangChain RAG* | Built retrieval pipeline with reranking and memory |
| *Evaluation Design* | Conducted quantitative and qualitative model comparisons |

---

## 🔍 Future Improvements

- 🔁 Add *DPO / RLAIF* for preference-based fine-tuning.  
- 🧾 Integrate *citation validation* metrics (e.g., factual overlap).  
- 🌍 Extend to *multilingual (English + Arabic)* biomedical QA.  
- 🧠 Enhance *RAG* with structured knowledge graphs (UMLS / SNOMED).  
- ☁ Deploy live demo via *Hugging Face Space* or *Docker* container.


⸻

🧪 Evaluation Snapshot


⸻
