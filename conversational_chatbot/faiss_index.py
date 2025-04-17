import os
import logging
from dotenv import load_dotenv
from google import genai
import faiss
import numpy as np
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API key and configuration from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./documents")  # Default to "./documents" if not set

if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY is not set. Please set it in your .env file.")
    raise ValueError("GOOGLE_API_KEY is not set. Please set it in your .env file.")

if not os.path.exists(DOCUMENTS_PATH):
    logging.error(f"DOCUMENTS_PATH '{DOCUMENTS_PATH}' does not exist.")
    raise ValueError(f"DOCUMENTS_PATH '{DOCUMENTS_PATH}' does not exist.")

# -----------------------------------------
# Gemini Embeddings class
# -----------------------------------------
# This class is used to generate embeddings using the Gemini API.
# It inherits from the LangChain Embeddings class.
# The embed_documents method takes a list of texts and returns their embeddings.
# The embed_query method takes a single text and returns its embedding.
# The __call__ method allows the class to be called like a function, returning the embedding of a single text.
# The class uses the Google Generative AI client to interact with the Gemini API.
# The embeddings are generated using the "embedding-001" model by default.
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        logging.info("🔄 Computing embeddings with Gemini...")
        embeddings = []
        for text in tqdm(texts, desc="Embedding chunks"):
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
                config={"task_type": "SEMANTIC_SIMILARITY"}
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text):
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config={"task_type": "SEMANTIC_SIMILARITY"}
        )
        return result.embeddings[0].values

    def __call__(self, text: str):
        return self.embed_query(text)
    

# Initialize Gemini embeddings
gemini = GeminiEmbeddings(api_key=GOOGLE_API_KEY)

def upload_htmls():
    """Loads HTML files, splits them into chunks, and creates a FAISS index."""
    try:
        logging.info(f"Loading documents from: {DOCUMENTS_PATH}")
        loader = DirectoryLoader(path=DOCUMENTS_PATH)
        documents = loader.load()
        logging.info(f"{len(documents)} Pages Loaded")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        )
        split_documents = text_splitter.split_documents(documents=documents)
        logging.info(f"Split into {len(split_documents)} Chunks...")

        texts = [doc.page_content for doc in split_documents]
        vectors = gemini.embed_documents(texts)

        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype("float32"))

        # Create FAISS database
        faiss_db = FAISS(
            embedding_function=gemini,
            index=index,
            docstore=dict(zip(range(len(split_documents)), split_documents)),
            index_to_docstore_id={i: i for i in range(len(split_documents))}
        )

        faiss_db.save_local("faiss_index")
        logging.info("FAISS index saved locally as 'faiss_index'.")
    except Exception as e:
        logging.error(f"Error during FAISS index creation: {e}")
        raise


def faiss_query():
    """Queries the FAISS index and retrieves relevant documents."""
    try:
        query = "Explain the Candidate Onboarding process."
        logging.info(f"Processing query: {query}")
        query_embedding = gemini.embed_query(query)

        db = FAISS.load_local("faiss_index", embeddings=gemini, allow_dangerous_deserialization=True)
        docs = db.similarity_search_by_vector(query_embedding)

        for doc in docs:
            logging.info("##---- Page ---##")
            logging.info(doc.metadata.get('source', 'N/A'))
            logging.info("##---- Content ---##")
            logging.info(doc.page_content)
    except Exception as e:
        logging.error(f"Error during FAISS query: {e}")
        raise


# Entry point
if __name__ == "__main__":
    # Uncomment the following line to create the FAISS index (run only once)
    # upload_htmls()

    # Query the FAISS index
    faiss_query()
