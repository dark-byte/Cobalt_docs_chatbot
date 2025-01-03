import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import logging
from datetime import datetime, UTC

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    url: str
    text: str
    score: float = 0.0

class DocsChatbot:
    def __init__(self):
        load_dotenv()
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize Pinecone
        pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = pinecone.Index('cobalt-docs')
        
        # Load documents
        self.docs_data = self._load_documents()
        
        self.embedding_model = 'text-embedding-ada-002'
        self.chat_model = 'gpt-3.5-turbo'

    def _load_documents(self) -> List[Dict]:
        try:
            with open('cobalt_docs_data/all_data.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("Documents file not found")
            return []
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in documents file")
            return []

    def get_query_embedding(self, query: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Document]:
        try:
            query_embedding = self.get_query_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            documents = []
            for result in results.get("matches", []):
                doc_text = self._get_document_text(result["metadata"]["url"])
                if doc_text:
                    documents.append(Document(
                        url=result["metadata"]["url"],
                        text=doc_text,
                        score=result["score"]
                    ))
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def _get_document_text(self, url: str) -> Optional[str]:
        for doc in self.docs_data:
            if doc['url'] == url:
                return doc['text']
        return None

    def generate_response(self, query: str, documents: List[Document]) -> str:
        try:
            context = self._build_context(documents)
            messages = [
                {
                    "role": "system",
                    "content": f"""You are a documentation assistant. Provide accurate, concise answers based on the context.
                    Always include relevant document URLs as references. If you're unsure or the context doesn't contain
                    relevant information, say so.\nContext:\n{context}\n\n"""
                },
                {
                    "role": "user",
                    "content": f"Query: {query}"
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response."

    def _build_context(self, documents: List[Document]) -> str:
        return "\n\n".join([
            f"[URL: {doc.url}] (Relevance: {doc.score:.2f})\n{doc.text}"
            for doc in documents
        ])

    def get_response(self, query: str) -> Dict:
        try:
            documents = self.retrieve_documents(query)
            if not documents:
                return {
                    "error": "No relevant documents found",
                    "timestamp": datetime.now(UTC).isoformat()
                }
            
            response = self.generate_response(query, documents)
            return {
                "response": response,
                "sources": [{"url": doc.url, "score": doc.score} for doc in documents],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": "Failed to process query",
                "timestamp": datetime.utcnow().isoformat()
            }

# Initialize chatbot
chatbot = DocsChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "error": "No query provided",
                "timestamp": datetime.utcnow().isoformat()
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "error": "Empty query provided",
                "timestamp": datetime.utcnow().isoformat()
            }), 400

        result = chatbot.get_response(query)
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unexpected error: {error}")
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.utcnow().isoformat()
    }), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)