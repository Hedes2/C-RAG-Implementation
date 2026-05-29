# 🔄 Corrective RAG (CRAG) — Ambiguous Query Handling

A **Corrective Retrieval-Augmented Generation** pipeline built with **LangGraph**, inspired by the research paper [**"Corrective Retrieval Augmented Generation"** (Yan et al., 2024)](https://arxiv.org/abs/2401.15884). It intelligently evaluates retrieved documents, classifies retrieval quality as **CORRECT**, **INCORRECT**, or **AMBIGUOUS**, and dynamically routes queries through knowledge refinement and web search to produce high-quality answers.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Research Paper](#research-paper)
- [Architecture](#architecture)
  - [Pipeline Flow](#pipeline-flow)
  - [Graph Nodes](#graph-nodes)
  - [Routing Logic](#routing-logic)
- [Tech Stack](#tech-stack)
- [Alternative / Substitute APIs & Tools](#alternative--substitute-apis--tools)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

Standard RAG pipelines blindly trust whatever documents the retriever returns, leading to poor answers when retrieval quality is low. **Corrective RAG** solves this by adding a **retrieval evaluator** that scores each retrieved chunk and routes the query through different paths depending on document relevance:

| Verdict | Condition | Action |
|---|---|---|
| ✅ **CORRECT** | At least one chunk scores **> 0.7** | Refine internal docs → Generate |
| ❌ **INCORRECT** | All chunks score **< 0.3** | Rewrite query → Web search → Refine → Generate |
| ⚠️ **AMBIGUOUS** | Scores between thresholds (no chunk > 0.7, but not all < 0.3) | Rewrite query → Web search → Merge internal + web docs → Refine → Generate |

This ensures the system **self-corrects** by supplementing weak retrievals with web search results before generating the final answer.

---

## Research Paper

This project is an implementation inspired by the following research paper:

> **Corrective Retrieval Augmented Generation**
> *Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling*
> Published: January 2024
> 📄 **Paper:** [https://arxiv.org/abs/2401.15884](https://arxiv.org/abs/2401.15884)

### Key Ideas from the Paper

The paper identifies a critical weakness in standard RAG systems — they **blindly trust** retrieved documents regardless of their relevance quality. CRAG addresses this with three core contributions:

1. **Retrieval Evaluator** — A lightweight evaluator assesses the overall quality of retrieved documents and assigns a confidence score. Based on this score, the system triggers different knowledge retrieval actions (**Correct**, **Incorrect**, or **Ambiguous**).

2. **Web Search Augmentation** — Since retrieval from static and limited corpora can only return sub-optimal documents, large-scale web searches are used as an extension to augment retrieval results when internal documents are insufficient.

3. **Decompose-then-Recompose** — A knowledge refinement algorithm decomposes retrieved documents into fine-grained knowledge strips (sentences), filters out irrelevant information, and recomposes only the relevant pieces for the final generation.

### How This Implementation Maps to the Paper

| Paper Concept | This Implementation |
|---|---|
| Retrieval Evaluator (confidence score) | `eval_each_doc_node()` — LLM scores each chunk `[0.0 – 1.0]` with `DocEvalScore` |
| Correct / Incorrect / Ambiguous actions | Conditional routing via `route_after_eval()` with `UPPER_TH=0.7` and `LOWER_TH=0.3` |
| Web Search Augmentation | `rewrite_query_node()` + `web_search_node()` using Tavily API |
| Decompose-then-Recompose | `refine()` — sentence-level decomposition + LLM `KeepOrDrop` filter |
| Plug-and-play design | Built as a modular LangGraph `StateGraph` with swappable components |

> **Note:** This is a simplified, educational implementation of the paper's ideas using LangGraph and Ollama. The original paper uses a fine-tuned T5-large model as the retrieval evaluator, whereas this implementation uses an LLM with structured output for evaluation.

---

## Architecture

### Pipeline Flow

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  retrieve   │   ← FAISS similarity search (k=4)
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │ eval_each_doc   │   ← LLM scores each chunk [0.0 – 1.0]
                    └──────┬──────────┘
                           │
                   ┌───────┼────────┐
                   │ route_after_eval│
                   └───┬───────┬────┘
                       │       │
            CORRECT    │       │  INCORRECT / AMBIGUOUS
                       │       │
                       │  ┌────▼────────┐
                       │  │rewrite_query│  ← LLM rewrites question for web search
                       │  └────┬────────┘
                       │       │
                       │  ┌────▼────────┐
                       │  │ web_search  │  ← Tavily API (top 5 results)
                       │  └────┬────────┘
                       │       │
                   ┌───▼───────▼───┐
                   │    refine     │   ← Sentence-level decomposition + LLM filter
                   └───────┬───────┘
                           │
                    ┌──────▼──────┐
                    │  generate   │   ← Final answer using refined context
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │     END     │
                    └─────────────┘
```

### Graph Nodes

| Node | Function | Description |
|---|---|---|
| `retrieve` | `retrieve_node()` | Queries FAISS vector store and returns top-k (4) similar document chunks |
| `eval_each_doc` | `eval_each_doc_node()` | Scores each retrieved chunk for relevance using structured LLM output (`DocEvalScore`). Classifies retrieval as CORRECT / INCORRECT / AMBIGUOUS |
| `rewrite_query` | `rewrite_query_node()` | Rewrites the user question into an optimized web search query (6–14 keyword tokens) using structured LLM output (`WebQuery`) |
| `web_search` | `web_search_node()` | Performs web search via Tavily API, returning top 5 results as LangChain `Document` objects |
| `refine` | `refine()` | Decomposes context into sentences, then uses an LLM filter (`KeepOrDrop`) to keep only sentences directly relevant to the question |
| `generate` | `generate()` | Generates the final answer using the refined context with an ML tutor system prompt |

### Routing Logic

The conditional edge after `eval_each_doc` determines the path:

```python
def route_after_eval(state: State) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"       # Skip web search, go directly to refinement
    else:
        return "rewrite_query"  # INCORRECT or AMBIGUOUS → web search path
```

**Key difference for AMBIGUOUS vs INCORRECT** happens inside the `refine` node:
- **CORRECT** → uses only `good_docs` (internal retrieval)
- **INCORRECT** → uses only `web_docs` (web search results)
- **AMBIGUOUS** → merges `good_docs + web_docs` (both sources)

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | [Ollama](https://ollama.com/) — `llama3.1` | Text generation, evaluation, and query rewriting |
| **Embeddings** | [Ollama](https://ollama.com/) — `nomic-embed-text` | Document embedding for vector similarity search |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) (via LangChain) | Fast similarity search over document chunks |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful graph-based pipeline with conditional routing |
| **Framework** | [LangChain](https://github.com/langchain-ai/langchain) | Document loaders, text splitters, prompt templates, structured output |
| **Web Search** | [Tavily Search API](https://tavily.com/) | Real-time web search for supplementing weak retrievals |
| **Document Loading** | `PyPDFLoader` (LangChain) | PDF document ingestion |
| **Text Splitting** | `RecursiveCharacterTextSplitter` | Chunking documents (900 chars, 150 overlap) |
| **Environment** | `python-dotenv` | Environment variable management |
| **Language** | Python 3.10+ | Core language |

---

## Alternative / Substitute APIs & Tools

If you don't want to use the exact tools in this project, here are drop-in alternatives:

### 🤖 LLM (instead of Ollama + Llama 3.1)

| Alternative | Package | Notes |
|---|---|---|
| **OpenAI GPT-4 / GPT-4o** | `langchain-openai` | `ChatOpenAI(model="gpt-4o")` — Best quality, paid API |
| **Google Gemini** | `langchain-google-genai` | `ChatGoogleGenerativeAI(model="gemini-2.0-flash")` — Free tier available |
| **Anthropic Claude** | `langchain-anthropic` | `ChatAnthropic(model="claude-sonnet-4-20250514")` — Excellent reasoning |
| **Groq (hosted Llama/Mixtral)** | `langchain-groq` | `ChatGroq(model="llama-3.1-70b-versatile")` — Free, very fast inference |
| **HuggingFace (local)** | `langchain-huggingface` | Run open-source models locally via Transformers |
| **Azure OpenAI** | `langchain-openai` | `AzureChatOpenAI(...)` — Enterprise-grade, same models as OpenAI |
| **AWS Bedrock** | `langchain-aws` | Access Claude, Llama, Titan via AWS |
| **Cohere** | `langchain-cohere` | `ChatCohere(model="command-r-plus")` — Good for RAG tasks |

### 📐 Embeddings (instead of Ollama + nomic-embed-text)

| Alternative | Package | Notes |
|---|---|---|
| **OpenAI Embeddings** | `langchain-openai` | `OpenAIEmbeddings(model="text-embedding-3-small")` — High quality, paid |
| **Google Embeddings** | `langchain-google-genai` | `GoogleGenerativeAIEmbeddings(model="models/embedding-001")` |
| **HuggingFace Embeddings** | `langchain-huggingface` | `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` — Free, local |
| **Cohere Embeddings** | `langchain-cohere` | `CohereEmbeddings(model="embed-english-v3.0")` — Free tier available |
| **AWS Bedrock Embeddings** | `langchain-aws` | Titan or Cohere embeddings via AWS |

### 🗄️ Vector Store (instead of FAISS)

| Alternative | Package | Notes |
|---|---|---|
| **ChromaDB** | `langchain-chroma` | Easy to use, persistent, no external server needed |
| **Pinecone** | `langchain-pinecone` | Fully managed cloud vector DB, scalable |
| **Weaviate** | `langchain-weaviate` | Open-source, supports hybrid search |
| **Qdrant** | `langchain-qdrant` | High-performance, open-source, supports filtering |
| **Milvus / Zilliz** | `langchain-milvus` | Scalable, cloud or self-hosted |
| **pgvector (PostgreSQL)** | `langchain-postgres` | Use your existing PostgreSQL with vector support |

### 🔍 Web Search (instead of Tavily)

| Alternative | Package / API | Notes |
|---|---|---|
| **Google Search (SerpAPI)** | `langchain-community` → `SerpAPIWrapper` | Requires `SERPAPI_API_KEY` |
| **Google Search (Serper)** | `langchain-community` → `GoogleSerperAPIWrapper` | Requires `SERPER_API_KEY`, cheaper |
| **DuckDuckGo** | `langchain-community` → `DuckDuckGoSearchResults` | **Free, no API key needed** |
| **Brave Search** | `langchain-community` → `BraveSearchWrapper` | Privacy-focused, free tier |
| **Bing Search** | `langchain-community` → `BingSearchAPIWrapper` | Requires Azure subscription |
| **Wikipedia** | `langchain-community` → `WikipediaAPIWrapper` | Free, good for factual queries |
| **You.com Search** | `langchain-community` → `YouSearchAPIWrapper` | AI-powered search |

### 📄 Document Loaders (instead of PyPDFLoader)

| Alternative | Package | Notes |
|---|---|---|
| **Unstructured** | `langchain-unstructured` | Handles PDF, DOCX, HTML, images, etc. |
| **PyMuPDF (fitz)** | `langchain-community` → `PyMuPDFLoader` | Faster PDF parsing |
| **PDFPlumber** | `langchain-community` → `PDFPlumberLoader` | Better table extraction |
| **Docx2txt** | `langchain-community` → `Docx2txtLoader` | For Word documents |
| **CSVLoader** | `langchain-community` → `CSVLoader` | For CSV/spreadsheet data |

---

## Project Structure

```
corrective-rag/
├── documents/                  # PDF documents for the knowledge base
│   ├── book1.pdf               # Primary ML/AI textbook (used in notebook)
│   ├── book2.pdf               # Additional reference material
│   └── book3.pdf               # Additional reference material
├── myenv/                      # Python virtual environment
├── 1_basic_rag.ipynb           # Step 1: Basic RAG pipeline
├── 2_retrieval_refinement.ipynb # Step 2: Adding retrieval refinement
├── 3_retrieval_evaluator.ipynb  # Step 3: LLM-based retrieval evaluation
├── 4_web_search_refinement.ipynb # Step 4: Web search fallback
├── 5_query_rewrite.ipynb        # Step 5: Query rewriting for web search
├── 6_ambiguous.ipynb            # Step 6: Full CRAG with AMBIGUOUS handling ⭐
├── .env                         # Environment variables (API keys)
└── README.md                    # This file
```

---

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running locally ([Download Ollama](https://ollama.com/download))
3. **Tavily API Key** (sign up at [tavily.com](https://tavily.com/) — free tier available)
4. Required Ollama models pulled:
   ```bash
   ollama pull llama3.1
   ollama pull nomic-embed-text
   ```

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/campusx-official/corrective-rag.git
   cd corrective-rag
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv myenv

   # Windows
   myenv\Scripts\activate

   # macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install langchain langchain-community langchain-core langchain-ollama
   pip install langgraph
   pip install faiss-cpu
   pip install pypdf
   pip install python-dotenv
   pip install tavily-python
   ```

   Or install all at once:
   ```bash
   pip install langchain langchain-community langchain-core langchain-ollama langgraph faiss-cpu pypdf python-dotenv tavily-python
   ```

4. **Place your PDF documents** in the `documents/` folder

5. **Make sure Ollama is running** with the required models:
   ```bash
   ollama serve           # Start Ollama server (if not already running)
   ollama pull llama3.1   # Download LLM
   ollama pull nomic-embed-text  # Download embedding model
   ```

---

## Environment Variables

Create a `.env` file in the project root:

```env
TAVILY_API_KEY=your_tavily_api_key_here
```

> **⚠️ Important:** Never commit your actual API keys to version control. Add `.env` to your `.gitignore`.

---

## Usage

1. Open `6_ambiguous.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially
3. Modify the query in the final cell to test different questions:

```python
res = app.invoke(
    {
        "question": "What is linear regression",  # ← Change this
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }
)

print("VERDICT:", res["verdict"])
print("REASON:", res["reason"])
print("WEB_QUERY:", res["web_query"])
print("\nOUTPUT:\n", res["answer"])
```

### Example Output

```
VERDICT: CORRECT
REASON: At least one retrieved chunk scored > 0.7.
WEB_QUERY:

OUTPUT:
 Linear regression is a type of model that predicts the value of one or more
 continuous target variables based on a linear combination of the input variables,
 where the relationship between inputs and outputs is modeled using a linear function.
```

---

## Configuration

Key parameters you can tune in the notebook:

| Parameter | Default | Description |
|---|---|---|
| `UPPER_TH` | `0.7` | Score threshold above which retrieval is considered **CORRECT** |
| `LOWER_TH` | `0.3` | Score threshold below which retrieval is considered **INCORRECT** |
| `chunk_size` | `900` | Character size for document chunks |
| `chunk_overlap` | `150` | Overlap between consecutive chunks |
| `search_kwargs["k"]` | `4` | Number of documents to retrieve from FAISS |
| `max_results` (Tavily) | `5` | Number of web search results to fetch |
| `temperature` (LLM) | `0.2` | LLM temperature (lower = more deterministic) |

---

## License

This project is open-source and available for educational purposes.

---

<p align="center">
  Built with ❤️ using LangChain, LangGraph & Ollama
</p>
