"""Run the real local two-tool workflow over the existing Pilot index."""
import argparse
from pathlib import Path

from app.agent.maintenance_agent import MaintenanceAgent
from app.rag.embedding import EmbeddingConfig, SentenceTransformerEmbeddingProvider
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.retriever import Retriever
from app.schemas.maintenance import MaintenanceContext
from app.tools.knowledge_tool import KnowledgeTool


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=root / "demo_data/raw_data1.xlsx")
    parser.add_argument("--qdrant-path", type=Path, default=root / "knowledge/vector_store/phase3_7b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.qdrant_path / "meta.json").is_file():
        parser.error("An existing Pilot index is required.")
    provider = SentenceTransformerEmbeddingProvider(EmbeddingConfig.from_env())
    with QdrantVectorStore(collection_name="semantic_validation", vector_size=provider.dimension,
                           path=args.qdrant_path, embedding_info=provider.info) as store:
        knowledge = KnowledgeTool(Retriever(provider, store))
        report = MaintenanceAgent(knowledge_tool=knowledge.retrieve_knowledge).run(MaintenanceContext(
            file_path=str(args.file), spm=90, equipment_type="drilling_pump",
            start_time="15:35:00", end_time="16:02:00", top_k=3))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    if report.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
