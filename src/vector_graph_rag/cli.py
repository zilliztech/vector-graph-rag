"""
Command-line interface for Vector Graph RAG.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from vector_graph_rag import VectorGraphRAG
from vector_graph_rag.config import Settings


def load_documents_from_file(file_path: str) -> list:
    """Load documents from a JSON or text file."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Handle different JSON formats
        if isinstance(data, list):
            if data and isinstance(data[0], str):
                return data
            elif data and isinstance(data[0], dict):
                # Assume format with "passage" and optional "triplets"
                if "passage" in data[0]:
                    return data
                elif "text" in data[0]:
                    return [d["text"] for d in data]
        
        raise ValueError("Unsupported JSON format. Expected list of strings or list of dicts with 'passage' or 'text' key.")
    
    else:
        # Treat as plain text file
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split by double newlines for multiple passages
        passages = [p.strip() for p in content.split("\n\n") if p.strip()]
        return passages


def cmd_index(args):
    """Index documents into the knowledge base."""
    print(f"Loading documents from: {args.input}")
    documents = load_documents_from_file(args.input)
    print(f"Loaded {len(documents)} documents")
    
    # Create RAG instance
    rag = VectorGraphRAG(
        milvus_uri=args.milvus_uri,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )
    
    # Check if documents have triplets
    if documents and isinstance(documents[0], dict) and "triplets" in documents[0]:
        print("Using pre-extracted triplets...")
        result = rag.add_documents_with_triplets(documents, show_progress=True)
    else:
        print("Extracting triplets using LLM...")
        result = rag.add_documents(documents, show_progress=True)
    
    print("\n" + "=" * 50)
    print("Indexing complete!")
    print(f"  Entities:  {len(result.entities)}")
    print(f"  Relations: {len(result.relations)}")
    print(f"  Passages:  {len(result.documents)}")
    print("=" * 50)


def cmd_query(args):
    """Query the knowledge base."""
    # Create RAG instance
    rag = VectorGraphRAG(
        milvus_uri=args.milvus_uri,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )
    
    question = args.question
    
    if args.interactive:
        print("Vector Graph RAG - Interactive Mode")
        print("Type 'quit' or 'exit' to exit")
        print("=" * 50)
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                if question.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nSearching...")
                result = rag.query(question, use_reranking=not args.no_rerank)
                
                print(f"\nAnswer: {result.answer}")
                
                if args.verbose:
                    print(f"\n[Retrieved Relations: {len(result.retrieved_relations)}]")
                    print(f"[Expanded Relations: {len(result.expanded_relations)}]")
                    print(f"[Reranked Relations: {len(result.reranked_relations)}]")
                    print(f"[Retrieved Passages: {len(result.retrieved_passages)}]")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    else:
        if not question:
            print("Error: Please provide a question with -q/--question or use -i/--interactive mode")
            sys.exit(1)
        
        print(f"Question: {question}\n")
        print("Searching...")
        
        result = rag.query(question, use_reranking=not args.no_rerank)
        
        print(f"\nAnswer: {result.answer}")
        
        if args.compare_naive:
            print("\n" + "-" * 50)
            print("Comparing with Naive RAG...")
            naive_result = rag.query_naive(question)
            print(f"Naive RAG Answer: {naive_result.answer}")
        
        if args.verbose:
            print("\n" + "=" * 50)
            print("Retrieval Details:")
            print(f"  Retrieved Relations: {len(result.retrieved_relations)}")
            print(f"  Expanded Relations: {len(result.expanded_relations)}")
            print(f"  Reranked Relations: {len(result.reranked_relations)}")
            print(f"  Retrieved Passages: {len(result.retrieved_passages)}")
            
            if result.reranked_relations:
                print("\nTop Reranked Relations:")
                for i, rel in enumerate(result.reranked_relations[:5], 1):
                    print(f"  {i}. {rel}")


def cmd_demo(args):
    """Run the demo with sample data."""
    print("=" * 60)
    print("  Vector Graph RAG Demo")
    print("=" * 60)
    
    # Sample data from the Bernoulli family
    nano_dataset = [
        {
            "passage": "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
            "triplets": [
                ["Jakob Bernoulli", "made significant contributions to", "calculus"],
                ["Jakob Bernoulli", "made significant contributions to", "the theory of probability"],
                ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
                ["Jakob Bernoulli", "is known for", "the Bernoulli theorem"],
                ["The Bernoulli theorem", "is a precursor to", "the law of large numbers"],
                ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
            ],
        },
        {
            "passage": "Johann Bernoulli (1667–1748): Johann, Jakob's younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
            "triplets": [
                ["Johann Bernoulli", "was a major figure of", "the development of calculus"],
                ["Johann Bernoulli", "was", "Jakob's younger brother"],
                ["Johann Bernoulli", "worked on", "infinitesimal calculus"],
                ["Johann Bernoulli", "was instrumental in spreading", "Leibniz's ideas"],
                ["Johann Bernoulli", "contributed to", "the calculus of variations"],
                ["Johann Bernoulli", "was known for", "the brachistochrone problem"],
            ],
        },
        {
            "passage": "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli's principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
            "triplets": [
                ["Daniel Bernoulli", "was the son of", "Johann Bernoulli"],
                ["Daniel Bernoulli", "made major contributions to", "fluid dynamics"],
                ["Daniel Bernoulli", "made major contributions to", "probability"],
                ["Daniel Bernoulli", "made major contributions to", "statistics"],
                ["Daniel Bernoulli", "is most famous for", "Bernoulli's principle"],
                ["Bernoulli's principle", "is fundamental to", "the understanding of aerodynamics"],
            ],
        },
        {
            "passage": "Leonhard Euler (1707–1783) was one of the greatest mathematicians of all time, and his relationship with the Bernoulli family was significant. Euler was born in Basel and was a student of Johann Bernoulli, who recognized his exceptional talent and mentored him in mathematics. Johann Bernoulli's influence on Euler was profound, and Euler later expanded upon many of the ideas and methods he learned from the Bernoullis.",
            "triplets": [
                ["Leonhard Euler", "had a significant relationship with", "the Bernoulli family"],
                ["Leonhard Euler", "was born in", "Basel"],
                ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
                ["Johann Bernoulli's influence", "was profound on", "Euler"],
            ],
        },
    ]
    
    print("\n📚 Loading sample data (Bernoulli family & Euler)...")
    
    # Create RAG instance
    rag = VectorGraphRAG(
        milvus_uri=args.milvus_uri or "./demo_vector_graph_rag.db",
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )
    
    # Add documents with pre-extracted triplets
    result = rag.add_documents_with_triplets(nano_dataset, show_progress=True)
    
    print(f"\n✅ Indexed: {len(result.entities)} entities, {len(result.relations)} relations, {len(result.documents)} passages")
    
    # Demo query
    demo_query = "What contribution did the son of Euler's teacher make?"
    
    print("\n" + "=" * 60)
    print(f"📝 Demo Query: {demo_query}")
    print("=" * 60)
    
    # Graph RAG
    print("\n🔍 Graph RAG Result:")
    graph_result = rag.query(demo_query)
    print(f"Answer: {graph_result.answer}")
    
    # Naive RAG for comparison
    print("\n🔍 Naive RAG Result (for comparison):")
    naive_result = rag.query_naive(demo_query)
    print(f"Answer: {naive_result.answer}")
    
    print("\n" + "=" * 60)
    print("✨ Demo complete! Notice how Graph RAG correctly identifies")
    print("   Daniel Bernoulli as the son of Euler's teacher (Johann Bernoulli),")
    print("   while Naive RAG may struggle with this multi-hop question.")
    print("=" * 60)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="vector-graph-rag",
        description="Vector Graph RAG - Graph RAG using pure vector search with Milvus",
    )
    
    # Common arguments
    parser.add_argument(
        "--milvus-uri",
        default="./vector_graph_rag.db",
        help="Milvus connection URI (default: ./vector_graph_rag.db)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model for extraction and generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model (default: text-embedding-3-small)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents into the knowledge base")
    index_parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input file (JSON or text)",
    )
    index_parser.set_defaults(func=cmd_index)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument(
        "-q", "--question",
        default=None,
        help="Question to answer",
    )
    query_parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    query_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show retrieval details",
    )
    query_parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking",
    )
    query_parser.add_argument(
        "--compare-naive",
        action="store_true",
        help="Compare with naive RAG",
    )
    query_parser.set_defaults(func=cmd_query)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo with sample data")
    demo_parser.set_defaults(func=cmd_demo)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
