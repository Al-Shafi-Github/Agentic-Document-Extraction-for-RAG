# Agentic Document Extraction (ADE) RAG Pipeline

## 📖 Overview

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline designed to handle complex business documents (specifically the Apple 10-K SEC filing).

Unlike traditional RAG pipelines that treat documents as flat text, this project leverages **Agentic Document Extraction (ADE)** to parse content into semantic chunks (text, tables, figures) with spatial metadata. This enables **Visual Grounding**—allowing users to not only get an answer but also see the exact location in the source document where the information resides.

## 🚀 Key Features

* **Semantic & Hybrid Search:** Combines vector similarity with metadata filtering (e.g., searching only "tables" for financial data).
* **Visual Grounding:** Uses Bounding Box (`bbox`) metadata to retrieve and display cropped images of the source evidence.
* **Structured Ingestion:** Handles ADE-parsed JSON and Markdown for precise context preservation.
* **Local Vector Store:** Utilizes **ChromaDB** with HNSW indexing for fast, persistent retrieval.
* **LangChain Integration:** modular chain architecture for the generation phase.

---

## 🏗️ Architecture

The pipeline follows the standard RAG triad, enhanced with ADE metadata:

1. **Preprocess:**
* Ingest ADE-parsed JSON (453 chunks from Apple's 10-K).
* Extract metadata: `page_number`, `chunk_type` (text/table/figure), and `bbox`.


2. **Retrieve:**
* Embed user queries using **OpenAI `text-embedding-3-small**`.
* Perform similarity search (Cosine/L2 distance) in ChromaDB.
* Apply **Hybrid Search** filters (e.g., `where chunk_type == 'table'`) for precision.


3. **Generate:**
* Retrieve top- relevant chunks.
* Pass structured context to an LLM via **LangChain**.
* Return the answer alongside visual evidence (cropped document images).



---

## 🛠️ Tech Stack

* **Language:** Python
* **Orchestration:** [LangChain](https://www.langchain.com/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** OpenAI `text-embedding-3-small` (1536 dimensions)
* **Data Source:** Agentic Document Extraction (ADE) output

---

## 📂 Data Structure

The pipeline expects pre-parsed data from ADE. The core data unit is a JSON object representing a document chunk:

```json
{
  "chunk_id": "unique_id_123",
  "chunk_type": "table",
  "text": "Total Net Sales ... $383,285",
  "bbox": [0.1, 0.25, 0.8, 0.45], 
  "page": 23
}

```

* **`bbox`**: Normalized coordinates `[x0, y0, x1, y1]` used for visual grounding.
* **`chunk_type`**: Critical for hybrid search (separating narrative text from financial tables).

---

## ⚡ Getting Started

### Prerequisites

* Python 3.9+
* An OpenAI API Key

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ade-rag-pipeline.git
cd ade-rag-pipeline

```

### 2. Install Dependencies

```bash
pip install langchain langchain-openai chromadb pandas

```

### 3. Environment Setup

Create a `.env` file or export your API key:

```bash
export OPENAI_API_KEY="your-sk-key-here"

```

### 4. Run the Pipeline

The main script performs ingestion, indexing, and a sample query.

```bash
python main.py

```

---

## 🔍 Usage Examples

### Basic Retrieval

The system uses LangChain's `create_retrieval_chain` to abstract the complexity:

```python
# Pseudo-code example
query = "What was the total revenue in 2023?"
response = rag_chain.invoke({"input": query})

print(response["answer"])
# Output: "Apple's total revenue for 2023 was $383.3 billion..."

```

### Hybrid Search (Metadata Filtering)

To reduce hallucinations on numerical queries, we can restrict search to tables:

```python
# Filtering within ChromaDB
results = vectorstore.similarity_search(
    "Net Sales by Category",
    k=3,
    filter={"chunk_type": "table"} 
)

```

### Visual Grounding

Since we hold `bbox` and `page` data, the system can render the source:

```python
def show_source(chunk):
    page_img = load_page_image(chunk.metadata['page'])
    crop = crop_image(page_img, chunk.metadata['bbox'])
    display(crop)

```

---

##  Future Improvements

* **Direct API Integration:** Replace static JSON loading with real-time calls to the ADE Parse API.
* **Multi-Modal RAG:** Pass the cropped images directly to a Vision-Language Model (like GPT-4o) for better table interpretation.
* **Evaluation:** Integrate RAGAS to measure context precision and answer faithfulness.

---

##  Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
