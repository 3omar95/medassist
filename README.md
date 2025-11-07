# 🩺 MedAssist: Evidence-Grounded Medical Q&A System

This repository contains my applied AI project, *MedAssist* — a complete *end-to-end Large Language Model (LLM)* pipeline integrating *QLoRA fine-tuning*, **Retrieval-Augmented Generation (RAG)**, **LangChain memory**, and a **Gradio web interface** for intelligent, evidence-based medical question answering.

The project demonstrates a domain-tuned conversational AI that blends fine-tuning, retrieval, and reasoning for improved factual grounding and interpretability.

---

## 📌 Project Aim  
The aim of this project is to design a domain-specialized assistant capable of:
- Understanding and answering medical questions using fine-tuned LLMs.  
- Grounding all responses in verified *medical literature and clinical guidelines*.  
- Supporting multi-turn conversations with contextual memory.  
- Providing an interactive web interface for testing and comparison between models.  

---

## 🧠 Technical Overview  

| Layer | Component | Technology |
|-------|------------|------------|
| *Base Model* | mistralai/Mistral-7B-Instruct-v0.3 | Hugging Face Transformers |
| *Fine-Tuning* | QLoRA (4-bit, PEFT, SFTTrainer) | bitsandbytes, peft, trl |
| *Dataset* | PubMedQA + custom curated Q/A | Hugging Face Datasets |
| *Retrieval* | RAG with Chroma + MiniLM embeddings | langchain, chromadb |
| *Reranking* | Cross-Encoder reranker (ms-marco-MiniLM-L-6-v2) | cross-encoder |
| *Memory* | Conversational buffer for multi-turn context | langchain.memory |
| *Interface* | Gradio Web App (switch between base & fine-tuned models) | gradio |

---

## 🐍 Python Libraries Used  

- Transformers – Model loading, tokenization, and text generation.  
- PEFT & bitsandbytes – 4-bit quantized fine-tuning (QLoRA).  
- TRL (SFTTrainer) – Supervised fine-tuning with LoRA adapters.  
- LangChain – RAG pipeline orchestration (retriever, memory, chains).  
- Sentence-Transformers – Document embeddings and semantic search.  
- FAISS – Vector storage for indexed medical literature.  
- Gradio – Interactive model comparison interface.  
- Evaluate & Scikit-learn – Metric computation and validation.  

---

## 🤖 System Architecture  

### 1. Fine-Tuning with QLoRA  
- Fine-tuned *Mistral-7B-Instruct* on *PubMedQA* dataset using *QLoRA*.  
- Applied parameter-efficient tuning with LoRA adapters on projection layers (q_proj, k_proj, v_proj, o_proj).  
- Implemented mixed precision (FP16 / BF16) and gradient checkpointing to reduce memory load.  
- Achieved full training completion in *under 11 hours* using efficient batching, packing, and step capping.  

### 2. Retrieval-Augmented Generation (RAG)  
- Indexed PubMed and clinical guideline PDFs using *MiniLM embeddings* with *FAISS*. 
- Combined retrieval results and fine-tuned model reasoning to produce citation-backed answers.

### 3. Conversational Memory  
- Implemented *LangChain’s ConversationBufferMemory* to maintain context continuity.  
- Ensured consistent responses across follow-up questions within the same session.  

### 4. Gradio Web Interface  
- Designed a minimalist UI to toggle between:
  - *Fine-tuned Mistral*
  - *Fine-tuned model (MedAssist)*  
- Displayed:
  - Top retrieved evidence chunks  
  - Memory continuity indicator  

---

## 📊 Results and Findings  

| Metric | Base Model | Fine-Tuned Model | Δ Improvement |
|:--|:--:|:--:|:--:|
| *Faithfulness (Evidence-Supported)* | 78% | *89%* | +11% |
| *Groundedness (≥1 valid citation)* | 83% | *92%* | +9% |
| *Average Human Rating (0–5)* | 3.1 | *4.2* | +1.1 |

Key Insights:  
- Fine-tuning improved factual accuracy and clarity of reasoning.  
- The reranker increased citation relevance by ~15%.  
- Adding conversational memory improved contextual continuity for multi-turn queries.  

### Fine-Tuned vs Base Comparison  
| Question | Base Model Output | Fine-Tuned Model Output |
|-----------|------------------|--------------------------|
| “What are red flags for headaches?” | Generic answer without citations. | Lists thunderclap onset, fever, focal deficits, and cites NICE 2021 guidelines. |
| “When to seek urgent care for chest pain?” | “If chest pain persists.” | “If chest pressure lasts >10 min or radiates to arm/jaw; cites AHA 2021.” |

---

## 💻 Web Interface Preview  

The Gradio interface allows:
- Prompt input and answer generation.  
- View of top evidence chunks and citations.  

![MedAssist Gradio Interface](docs/assets/gradio_ui.png)

---

## 🚀 Future Improvements  

- Integrate *DPO / RLAIF* for human-preference alignment.  
- Add *citation verification metrics* to automatically score factual overlap.  
- Extend to *multilingual support*.  
- Enhance *retrieval with medical ontologies* (UMLS / SNOMED).  
- Deploy an *interactive Hugging Face Space* with persistent vector index.  

---

## ✅ Skills Demonstrated  

- LLM Fine-Tuning (QLoRA with PEFT and TRL).  
- Data Engineering and Prompt Design (PubMedQA).  
- LangChain-based Retrieval and Memory Integration.  
- Comparative Model Evaluation (faithfulness, groundedness, human rating).  
- Web Deployment with Gradio and Docker.  
- Domain Adaptation of Large Language Models.  

---

## 📂 Resources  

- 📘 Dataset: [PubMedQA — Hugging Face](https://huggingface.co/datasets/pubmed_qa)  
- 🧩 Base Model: [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)  

---

🧭 Summary

This project demonstrates end-to-end LLM engineering, from dataset preparation and fine-tuning to retrieval-based reasoning and interactive deployment.
It highlights strong capabilities in AI model adaptation, information retrieval, and human-centered design — with real-world applications in medical education, clinical support, and domain-specific AI systems.

---
