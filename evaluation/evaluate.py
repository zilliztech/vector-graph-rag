"""
Evaluation script for Vector Graph RAG.

Compares Vector Graph RAG with Naive RAG on multi-hop QA datasets:
- 2WikiMultiHopQA
- HotpotQA
- MuSiQue
"""

import json
import argparse
import os
import sys
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

from tqdm import tqdm

def setup_logging(log_dir: str = "logs", dataset_name: str = "eval") -> str:
    """
    Setup logging to both console and file.

    Args:
        log_dir: Directory to save log files
        dataset_name: Dataset name for log file naming

    Returns:
        Path to the log file
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{dataset_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler - saves all logs
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - for tqdm compatibility, only show warnings and errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return log_path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vector_graph_rag import VectorGraphRAG, Settings
from vector_graph_rag.models import Document, Triplet
from vector_graph_rag.llm.extractor import processing_phrases


def load_dataset(
    dataset_name: str, data_dir: str = "data"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load dataset and corpus.

    Args:
        dataset_name: Name of the dataset (2wikimultihopqa, hotpotqa, musique)
        data_dir: Directory containing data files

    Returns:
        Tuple of (questions, corpus)
    """
    questions_path = os.path.join(data_dir, f"{dataset_name}.json")
    corpus_path = os.path.join(data_dir, f"{dataset_name}_corpus.json")

    with open(questions_path, "r") as f:
        questions = json.load(f)

    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    return questions, corpus


def load_extracted_triplets(dataset_name: str, data_dir: str = "data") -> List[Dict]:
    """
    Load pre-extracted triplets from OpenIE output.

    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing data files

    Returns:
        List of documents with extracted triplets
    """
    # Find the extraction file
    extraction_pattern = f"openie_{dataset_name}_results_ner_gpt-3.5-turbo-1106_*.json"

    import glob

    files = glob.glob(os.path.join(data_dir, extraction_pattern))

    if not files:
        raise FileNotFoundError(
            f"No extraction file found matching {extraction_pattern}"
        )

    # Use the one with the most samples
    extraction_file = max(
        files, key=lambda x: int(x.split("_")[-1].replace(".json", ""))
    )

    print(f"Loading triplets from: {extraction_file}")

    with open(extraction_file, "r") as f:
        data = json.load(f)

    return data["docs"]


def build_documents_from_triplets(
    extracted_docs: List[Dict],
) -> List[Dict]:
    """
    Build document dicts from extracted triplets.

    Args:
        extracted_docs: List of dicts with 'passage', 'extracted_entities', 'extracted_triples'

    Returns:
        List of dicts with 'passage' and 'triplets' for VectorGraphRAG
    """
    documents = []

    for doc in extracted_docs:
        passage = doc.get("passage", "")
        raw_triplets = doc.get("extracted_triples", [])

        # Process triplets
        triplets = []
        for triple in raw_triplets:
            if isinstance(triple, list) and len(triple) == 3:
                # Normalize/preprocess
                subject = processing_phrases(str(triple[0]))
                predicate = processing_phrases(str(triple[1]))
                obj = processing_phrases(str(triple[2]))

                if subject and predicate and obj:
                    triplets.append([subject, predicate, obj])

        documents.append(
            {
                "passage": passage,
                "triplets": triplets,
            }
        )

    return documents


def calculate_recall(
    gold_items: set,
    retrieved_items: List[str],
    k_list: List[int] = [1, 2, 5, 10, 15, 20],
) -> Dict[int, float]:
    """
    Calculate recall at different k values.

    Args:
        gold_items: Set of gold passage titles/ids
        retrieved_items: List of retrieved passage titles/ids (ordered by rank)
        k_list: List of k values to compute recall at

    Returns:
        Dict mapping k to recall@k
    """
    recall = {}
    for k in k_list:
        hits = sum(1 for item in retrieved_items[:k] if item in gold_items)
        recall[k] = hits / len(gold_items) if gold_items else 0.0
    return recall


def get_gold_items(sample: Dict, dataset_name: str) -> set:
    """
    Get gold passage titles from a sample.
    """
    if dataset_name in ["hotpotqa", "2wikimultihopqa", "test_sample"]:
        gold_passages = sample.get("supporting_facts", [])
        return set(item[0] for item in gold_passages)
    elif dataset_name == "musique":
        gold_passages = [
            p for p in sample.get("paragraphs", []) if p.get("is_supporting")
        ]
        return set(p["title"] + "\n" + p["paragraph_text"] for p in gold_passages)
    else:
        gold_passages = [
            p for p in sample.get("paragraphs", []) if p.get("is_supporting")
        ]
        return set(p["title"] + "\n" + p["text"] for p in gold_passages)


def get_retrieved_titles(
    passages: List[str],
    dataset_name: str,
) -> List[str]:
    """
    Extract titles from retrieved passages.
    """
    if dataset_name in ["hotpotqa", "2wikimultihopqa", "test_sample"]:
        return [p.split("\n")[0].strip() for p in passages]
    else:
        return passages


class VectorGraphRAGEvaluator:
    """
    Evaluator for Vector Graph RAG.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str = "data",
        milvus_uri: Optional[str] = None,
        milvus_db: Optional[str] = None,
        milvus_index_type: Optional[str] = None,
        milvus_consistency_level: Optional[str] = None,
        use_pre_extracted: bool = True,
        top_k: int = 10,
        entity_top_k: Optional[int] = None,
        relation_top_k: Optional[int] = None,
        entity_similarity_threshold: Optional[float] = None,
        relation_similarity_threshold: Optional[float] = None,
        relation_number_threshold: Optional[int] = None,
        llm_model: Optional[str] = None,
        use_llm_cache: bool = True,
        embedding_model: Optional[str] = None,
        embedding_instruction: Optional[str] = None,
        embedding_instruction_template: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            dataset_name: Name of the dataset
            data_dir: Directory containing data files
            milvus_uri: Milvus connection URI
            milvus_db: Milvus database name
            milvus_index_type: Milvus index type (e.g., FLAT, AUTOINDEX)
            milvus_consistency_level: Milvus consistency level (e.g., Strong, Bounded)
            use_pre_extracted: Whether to use pre-extracted triplets
            top_k: Number of passages to retrieve
            entity_top_k: Number of top entities to retrieve (default: 10)
            relation_top_k: Number of top relations to retrieve (default: 10)
            entity_similarity_threshold: Similarity threshold for entity retrieval (default: 0.9)
            relation_similarity_threshold: Similarity threshold for relation retrieval (default: -1)
            relation_number_threshold: Max expanded relations, eviction if exceeded (default: 1000)
            llm_model: LLM model for reranking (default: gpt-4o-mini)
            use_llm_cache: Whether to use LLM response caching (default: True)
            embedding_model: Embedding model to use (e.g., facebook/contriever, text-embedding-3-large)
            embedding_instruction: Instruction for embedding model (for BGE/Qwen3 style models)
            embedding_instruction_template: Instruction template style (bge or qwen3)
        """
        self.embedding_model = embedding_model
        self.embedding_instruction = embedding_instruction
        self.embedding_instruction_template = embedding_instruction_template
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.top_k = top_k
        self.use_pre_extracted = use_pre_extracted

        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        self.questions, self.corpus = load_dataset(dataset_name, data_dir)
        print(
            f"Loaded {len(self.questions)} questions, {len(self.corpus)} corpus passages"
        )

        # Initialize RAG with settings
        self.milvus_uri = milvus_uri or f"./eval_{dataset_name}.db"
        # Ensure collection prefix starts with letter (Milvus requirement)
        collection_prefix = f"ds_{dataset_name}"
        settings = Settings(
            milvus_uri=self.milvus_uri,
            collection_prefix=collection_prefix,  # Use dataset name as collection prefix
        )
        if milvus_db:
            settings.milvus_db = milvus_db
        if milvus_index_type:
            settings.milvus_index_type = milvus_index_type
        if milvus_consistency_level:
            settings.milvus_consistency_level = milvus_consistency_level
        if entity_top_k is not None:
            settings.entity_top_k = entity_top_k
        if relation_top_k is not None:
            settings.relation_top_k = relation_top_k
        if entity_similarity_threshold is not None:
            settings.entity_similarity_threshold = entity_similarity_threshold
        if relation_similarity_threshold is not None:
            settings.relation_similarity_threshold = relation_similarity_threshold
        if relation_number_threshold is not None:
            settings.relation_number_threshold = relation_number_threshold
        if llm_model is not None:
            settings.llm_model = llm_model
        settings.use_llm_cache = use_llm_cache
        if embedding_model is not None:
            settings.embedding_model = embedding_model

        # Apply monkey-patching for embedding instruction if specified
        if embedding_instruction or embedding_instruction_template:
            from vector_graph_rag.storage.embeddings import EmbeddingModel
            original_embedding_init = EmbeddingModel.__init__

            def patched_embedding_init(self_emb, settings=None, model=None, instruction=None, instruction_template=None):
                actual_instruction = instruction if instruction is not None else embedding_instruction
                actual_template = instruction_template if instruction_template is not None else embedding_instruction_template
                original_embedding_init(self_emb, settings, model, actual_instruction, actual_template)

            EmbeddingModel.__init__ = patched_embedding_init
            self._original_embedding_init = original_embedding_init
        else:
            self._original_embedding_init = None

        self.rag = VectorGraphRAG(settings=settings)

    def has_existing_index(self) -> bool:
        """
        Check if collections already exist and have data in Milvus.
        """
        try:
            # Check if collections exist by trying to query them
            client = self.rag._store.client
            entity_col = self.rag._store.entity_collection
            passage_col = self.rag._store.passage_collection

            if not client.has_collection(entity_col) or not client.has_collection(passage_col):
                return False

            # Try to get at least one record from each collection
            entity_results = client.query(
                collection_name=entity_col,
                filter="",
                limit=1,
                output_fields=["id"],
            )
            passage_results = client.query(
                collection_name=passage_col,
                filter="",
                limit=1,
                output_fields=["id"],
            )
            return len(entity_results) > 0 and len(passage_results) > 0
        except Exception:
            return False

    def build_index(self, force_reindex: bool = False):
        """
        Build the index from corpus.

        Args:
            force_reindex: Whether to force re-indexing even if collections exist
        """
        # Check if index already exists in Milvus
        if not force_reindex and self.has_existing_index():
            print(f"Using existing index in Milvus (collections exist with data)")
            print("(Use --force-reindex to rebuild)")
            return

        # Need to build index - load data and insert
        print("Building index...")

        if self.use_pre_extracted:
            print("Loading pre-extracted triplets...")
            extracted_docs = load_extracted_triplets(self.dataset_name, self.data_dir)
            documents = build_documents_from_triplets(extracted_docs)
            print(f"Built {len(documents)} documents from triplets")
            self.rag.add_documents_with_triplets(documents, show_progress=True)
        else:
            passages = []
            for item in self.corpus:
                if isinstance(item, dict):
                    passage = item.get("title", "") + "\n" + item.get("text", "")
                else:
                    passage = str(item)
                passages.append(passage)
            print(f"Auto triplet extraction for {len(passages)} passages...")
            self.rag.add_documents(passages, show_progress=True)

        # Get stats from extraction result
        stats = self.rag.get_stats()
        print(f"Index built: entities={stats['entities']}, relations={stats['relations']}, passages={stats['passages']}")

    def evaluate(
        self,
        max_samples: Optional[int] = None,
        use_reranking: bool = True,
        k_list: List[int] = [1, 2, 5, 10, 15, 20],
        method: str = "both",
    ) -> Dict[str, Any]:
        """
        Run evaluation.

        Args:
            max_samples: Maximum number of samples to evaluate
            use_reranking: Whether to use LLM reranking
            k_list: List of k values for recall computation
            method: Retrieval method - 'both', 'graph', or 'naive'

        Returns:
            Dict with evaluation results
        """
        samples = self.questions[:max_samples] if max_samples else self.questions

        run_graph = method in ["both", "graph"]
        run_naive = method in ["both", "naive"]

        # Results tracking
        total_recall_graph = {k: 0.0 for k in k_list}
        total_recall_naive = {k: 0.0 for k in k_list}

        results = []

        for idx, sample in tqdm(
            enumerate(samples), total=len(samples), desc=f"Evaluating ({method})"
        ):
            question = sample["question"]
            sample_id = sample.get("_id", sample.get("id", idx))

            # Get gold items
            gold_items = get_gold_items(sample, self.dataset_name)

            if not gold_items:
                continue

            graph_titles = []
            naive_titles = []
            recall_graph = {k: 0.0 for k in k_list}
            recall_naive = {k: 0.0 for k in k_list}

            # Graph RAG retrieval (no answer generation)
            if run_graph:
                try:
                    graph_result = self.rag.retrieve(
                        question, use_reranking=use_reranking, top_k=self.top_k
                    )
                    graph_passages = graph_result.retrieved_passages
                    graph_titles = get_retrieved_titles(
                        graph_passages, self.dataset_name
                    )
                except Exception as e:
                    print(f"Error in Graph RAG for sample {sample_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    graph_titles = []
                recall_graph = calculate_recall(gold_items, graph_titles, k_list)

            # Naive RAG retrieval (no answer generation)
            if run_naive:
                try:
                    naive_result = self.rag.retrieve_naive(question, top_k=self.top_k)
                    naive_passages = naive_result.retrieved_passages
                    naive_titles = get_retrieved_titles(
                        naive_passages, self.dataset_name
                    )
                except Exception as e:
                    print(f"Error in Naive RAG for sample {sample_id}: {e}")
                    naive_titles = []
                recall_naive = calculate_recall(gold_items, naive_titles, k_list)

            for k in k_list:
                total_recall_graph[k] += recall_graph[k]
                total_recall_naive[k] += recall_naive[k]

            result_entry = {
                "id": sample_id,
                "question": question,
                "gold_items": list(gold_items),
            }
            if run_graph:
                result_entry["graph_retrieved"] = graph_titles[:5]
                result_entry["recall_graph"] = recall_graph
            if run_naive:
                result_entry["naive_retrieved"] = naive_titles[:5]
                result_entry["recall_naive"] = recall_naive

            results.append(result_entry)

            # Print and log progress
            if (idx + 1) % 1 == 0:
                n = idx + 1
                log_lines = []
                if run_graph:
                    graph_line = f"[{n}] Graph RAG: " + " ".join(
                        f"R@{k}={total_recall_graph[k]/n:.4f}" for k in k_list[:4]
                    )
                    log_lines.append(graph_line)
                    print(f"\n{graph_line}", end="")
                if run_naive:
                    naive_line = f"[{n}] Naive RAG: " + " ".join(
                        f"R@{k}={total_recall_naive[k]/n:.4f}" for k in k_list[:4]
                    )
                    log_lines.append(naive_line)
                    print(f"\n{naive_line}", end="")
                print()

                # Log to file
                for line in log_lines:
                    logging.info(line)

        # Final results
        n = len(results)
        final_recall_graph = (
            {k: total_recall_graph[k] / n for k in k_list} if run_graph else {}
        )
        final_recall_naive = (
            {k: total_recall_naive[k] / n for k in k_list} if run_naive else {}
        )

        return {
            "dataset": self.dataset_name,
            "num_samples": n,
            "method": method,
            "recall_graph_rag": final_recall_graph,
            "recall_naive_rag": final_recall_naive,
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vector Graph RAG")
    parser.add_argument(
        "--dataset",
        type=str,
        default="2wikimultihopqa",
        choices=["2wikimultihopqa", "hotpotqa", "musique", "test_sample"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="evaluation/data",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default=None,
        help="Milvus connection URI (e.g., http://localhost:19530)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Milvus database name",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force re-indexing even if collections already exist",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default=None,
        help="Milvus index type (e.g., FLAT, AUTOINDEX, IVF_FLAT, HNSW)",
    )
    parser.add_argument(
        "--consistency-level",
        type=str,
        default=None,
        help="Milvus consistency level (Strong, Bounded, Session, Eventually)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable LLM reranking",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of passages to retrieve",
    )
    parser.add_argument(
        "--entity-top-k",
        type=int,
        default=20,
        help="Number of top entities to retrieve (default: 20)",
    )
    parser.add_argument(
        "--relation-top-k",
        type=int,
        default=20,
        help="Number of top relations to retrieve (default: 20)",
    )
    parser.add_argument(
        "--entity-similarity-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for entity retrieval, keep if score > threshold (default: 0.9)",
    )
    parser.add_argument(
        "--relation-similarity-threshold",
        type=float,
        default=-1.0,
        help="Similarity threshold for relation retrieval, keep if score > threshold (default: -1, keeps all)",
    )
    parser.add_argument(
        "--relation-number-threshold",
        type=int,
        default=1000,
        help="Maximum number of expanded relations. If exceeded, use eviction strategy (default: 1000)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["both", "graph", "naive"],
        default="both",
        help="Retrieval method: 'both' (default), 'graph' (Graph RAG only), 'naive' (Naive RAG only)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to save log files (default: logs)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for reranking (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-llm-cache",
        action="store_true",
        help="Disable LLM response caching",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model to use (default: BAAI/bge-large-en-v1.5, or text-embedding-3-large for OpenAI)",
    )
    parser.add_argument(
        "--embedding-instruction",
        type=str,
        default=None,
        help="Instruction for embedding model (auto-set for BGE models if not specified)",
    )
    parser.add_argument(
        "--embedding-instruction-template",
        type=str,
        choices=["bge", "qwen3"],
        default=None,
        help="Instruction template style (auto-set for BGE models if not specified)",
    )

    args = parser.parse_args()

    # Auto-set BGE instruction if using BGE model and no instruction specified
    if args.embedding_model and "bge" in args.embedding_model.lower():
        if args.embedding_instruction is None:
            args.embedding_instruction = "Represent this sentence for searching relevant passages"
        if args.embedding_instruction_template is None:
            args.embedding_instruction_template = "bge"

    # Setup logging
    log_path = setup_logging(log_dir=args.log_dir, dataset_name=args.dataset)
    print(f"Logging to: {log_path}")
    logging.info(f"Starting evaluation for dataset: {args.dataset}")
    logging.info(f"Arguments: {vars(args)}")

    # Print embedding model info
    print(f"Using embedding model: {args.embedding_model}")
    if args.embedding_instruction:
        print(f"Embedding instruction: {args.embedding_instruction}")
    logging.info(f"Embedding model: {args.embedding_model}")
    if args.embedding_instruction:
        logging.info(f"Embedding instruction: {args.embedding_instruction}")

    # Create evaluator
    evaluator = VectorGraphRAGEvaluator(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        milvus_uri=args.milvus_uri,
        milvus_db=args.db,
        milvus_index_type=args.index_type,
        milvus_consistency_level=args.consistency_level,
        top_k=args.top_k,
        entity_top_k=args.entity_top_k,
        relation_top_k=args.relation_top_k,
        entity_similarity_threshold=args.entity_similarity_threshold,
        relation_similarity_threshold=args.relation_similarity_threshold,
        relation_number_threshold=args.relation_number_threshold,
        llm_model=args.llm_model,
        use_llm_cache=not args.no_llm_cache,
        embedding_model=args.embedding_model,
        embedding_instruction=args.embedding_instruction,
        embedding_instruction_template=args.embedding_instruction_template,
    )

    # Build index
    evaluator.build_index(force_reindex=args.force_reindex)

    # Run evaluation
    results = evaluator.evaluate(
        max_samples=args.max_samples,
        use_reranking=not args.no_rerank,
        method=args.method,
    )

    # Print and log results
    header = "=" * 60
    title = f"Evaluation Results - {args.dataset} (method: {args.method})"
    print(f"\n{header}")
    print(title)
    print(header)
    logging.info(header)
    logging.info(title)
    logging.info(header)

    if args.method in ["both", "graph"]:
        print("\nGraph RAG Recall:")
        logging.info("Graph RAG Recall:")
        for k, v in results["recall_graph_rag"].items():
            line = f"  R@{k}: {v:.4f}"
            print(line)
            logging.info(line)

    if args.method in ["both", "naive"]:
        print("\nNaive RAG Recall:")
        logging.info("Naive RAG Recall:")
        for k, v in results["recall_naive_rag"].items():
            line = f"  R@{k}: {v:.4f}"
            print(line)
            logging.info(line)

    if args.method == "both":
        print("\nImprovement (Graph RAG - Naive RAG):")
        logging.info("Improvement (Graph RAG - Naive RAG):")
        for k in results["recall_graph_rag"].keys():
            diff = results["recall_graph_rag"][k] - results["recall_naive_rag"][k]
            line = f"  R@{k}: {diff:+.4f}"
            print(line)
            logging.info(line)

    # Save results
    output_file = args.output or f"output/eval_results_{args.dataset}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Log saved to: {log_path}")
    logging.info(f"Results saved to: {output_file}")
    logging.info(f"Evaluation completed successfully")


if __name__ == "__main__":
    main()
