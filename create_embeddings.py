import os
import json
from openai import OpenAI

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# Load API keys
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')  # e.g., 'aws-us-west-2'

# Initialize Pinecone instance
pinecone = Pinecone(api_key=pinecone_api_key)

# Define index name and embedding model
index_name = 'cobalt-docs'
embedding_model = 'text-embedding-ada-002'

# Check if the index exists; if not, create it
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1536,  # 'text-embedding-ada-002' outputs 1536-dimensional vectors
        metric='cosine',  # Use cosine similarity
        spec=ServerlessSpec(
            cloud='aws',  # Replace with your preferred cloud provider if needed
            region=pinecone_environment
        )
    )

# Connect to the index
index = pinecone.Index(index_name)

# Load data from JSON file
with open('cobalt_docs_data/all_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to generate embeddings
def get_embedding(text):
    response = client.embeddings.create(input=[text], model=embedding_model)
    return response.data[0].embedding

# Upsert embeddings into Pinecone
batch_size = 100  # Number of embeddings to upsert in a single request
for i in tqdm(range(0, len(data), batch_size)):
    # Create a batch of embeddings
    batch = data[i:i + batch_size]
    ids = [item['url'] for item in batch]  # Use URLs as unique IDs
    texts = [item['text'] for item in batch]
    embeddings = [get_embedding(text) for text in texts]

    # Create list of (id, vector, metadata) tuples
    vectors = [(ids[j], embeddings[j], {'url': ids[j]}) for j in range(len(ids))]

    # Upsert vectors into Pinecone
    index.upsert(vectors)

print("Data successfully upserted into Pinecone.")
