import glob
import click
from chromadb import PersistentClient

from .config import load_settings
from .embed import Embedder
from .rag import RAGPipeline

@click.group()
def cli():
    pass

@cli.command()
def index():
    """Index all repo files into ChromaDB."""
    settings = load_settings()
    embedder = Embedder(api_key=settings.gemini_api_key)
    client = PersistentClient(path="./chroma")
    collection = client.get_or_create_collection("repo_docs")

    # Clear old docs
    try:
        collection.delete(where={})
    except Exception:
        pass

    click.echo("🔎 Scanning repo files...")
    files = glob.glob("./**/*.py", recursive=True)

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        emb = embedder.embed(content)
        collection.add(
            documents=[content],
            embeddings=[emb],
            ids=[file]
        )
        click.echo(f"✅ Indexed {file}")

    click.echo("🎉 Finished indexing repo!")

@cli.command()
@click.argument("question")
def query(question):
    """Ask a question about the repo."""
    settings = load_settings()
    embedder = Embedder(api_key=settings.gemini_api_key)
    rag = RAGPipeline(embedder)

    click.echo(f"❓ Question: {question}")
    answer = rag.query(question)
    click.echo("\n💡 Answer:\n")
    click.echo(answer)

@cli.command(name="generate-readme")
def generate_readme():
    """Generate or update README.md using repo context."""
    settings = load_settings()
    embedder = Embedder(api_key=settings.gemini_api_key)
    rag = RAGPipeline(embedder)

    question = """Generate a professional README.md for this repository.
It should include:
- Project Overview
- Key Features
- Installation instructions
- Usage examples
- Tech stack (if clear from context)
- Contributing & License (generic if not in repo)

Keep it clean, markdown-formatted, and concise."""

    click.echo("📝 Generating README.md ...")
    answer = rag.query(question)

    with open("README.md", "w", encoding="utf-8") as f:
        f.write(answer)

    click.echo("✅ README.md created/updated!")

if __name__ == "__main__":
    cli()
