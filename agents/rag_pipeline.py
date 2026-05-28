from pathlib import Path
from langchain_neo4j import Neo4jGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader

from config import (
    REGULATORY_DOCS_DIR,
    VECTOR_DB_PATH,
    GOOGLE_API_KEY,
    # EMBEDDING_MODEL
)

from utils.logger import get_logger

logger = get_logger("rag_pipeline")


class RegulatoryRAG:

    def __init__(self):

        self.embedding_model = (
            HuggingFaceEmbeddings(

                model_name="BAAI/bge-small-en-v1.5",

                model_kwargs={
                    "device": "cpu"
                },

                encode_kwargs={
                    "normalize_embeddings": True
                }
            )
        )
    # =====================================================
    # LOAD DOCUMENTS
    # =====================================================

    def load_documents(self):
        logger.info("Loading regulatory PDFs")

        pdf_files = list(Path(REGULATORY_DOCS_DIR).glob("*.pdf"))

        documents = []

        for pdf in pdf_files:
            logger.info(f"Loading PDF -> {pdf.name}")
            loader = PyPDFLoader(str(pdf))
            docs = loader.load()
            documents.extend(docs)

        logger.info(f"Loaded {len(documents)} pages")
        return documents

    # =====================================================
    # CHUNK DOCUMENTS
    # =====================================================
    def chunk_documents(self, documents):
        logger.info("Chunking documents")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} chunks")
        return chunks

    # =====================================================
    # BUILD VECTOR STORE (STABLE VERSION)
    # =====================================================
    def build_vectorstore(self):
        logger.info("Building vector database")

        documents = self.load_documents()

        if len(documents) == 0:
            raise Exception("No regulatory PDFs found")

        chunks = self.chunk_documents(documents)

        # --- STABILITY FIX START ---
        # Extract texts and metadatas separately for FAISS.from_texts
        # This prevents serialization errors with Gemini embeddings
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        # --- STABILITY FIX END ---

        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )

        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(VECTOR_DB_PATH))

        logger.info("Vector DB created successfully")
        return vector_store

    # =====================================================
    # LOAD VECTOR STORE
    # =====================================================
    def load_vectorstore(self):
        logger.info("Loading vector database")

        vector_store = FAISS.load_local(
            str(VECTOR_DB_PATH),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        logger.info("Vector DB loaded successfully")
        return vector_store

    # =====================================================
    # BUILD OR LOAD
    # =====================================================
    def build_or_load_vectorstore(self):
        faiss_index = VECTOR_DB_PATH / "index.faiss"

        if faiss_index.exists():
            logger.info("Using existing vector DB")
            return self.load_vectorstore()

        logger.info("Creating new vector DB")
        return self.build_vectorstore()

    # =====================================================
    # RETRIEVE CONTEXT
    # =====================================================
    def retrieve_context(self, vector_store, query, k=4):
        logger.info(f"Retrieving context -> {query}")

        docs = vector_store.similarity_search(query, k=k)

        context = "\n\n".join([doc.page_content for doc in docs])
        return context
