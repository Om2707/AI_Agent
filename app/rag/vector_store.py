import json
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def load_past_projects():
    """Load past project specifications from a JSON file."""
    with open(Path(__file__).parent / "past_projects.json", "r") as f:
        return json.load(f)

def create_vector_store():
    """Create and return a Qdrant vector store with past project specifications."""
    past_projects = load_past_projects()
    docs = []
    for project in past_projects:
        text = f"Project on {project['platform']} for {project['goal']} in {project['stage']} stage with title '{project.get('title', '')}' and overview '{project.get('overview', '')}'"
        doc = Document(page_content=text, metadata=project)
        docs.append(doc)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Qdrant.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",  # Local Qdrant instance
        collection_name="project_specs"
    )
    return vector_store

vector_store = create_vector_store()
__all__ = ["vector_store"]
