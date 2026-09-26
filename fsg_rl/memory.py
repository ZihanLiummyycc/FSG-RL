"""Persistent hybrid retrieval and structured FSG-RL memory updates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .api_client import ChatAPIConfig, OpenAICompatibleChatClient
from .datasets import load_json_records
from .schemas import FunctionGraph, KnowledgeItem, MemoryContext, Problem


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|[\u4e00-\u9fff]", text.lower()))


class MemoryBank:
    """Four memory stores with keyword, embedding, and optional API reranking."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        section = config.get("memory", {})
        self.top_k = int(section.get("top_k", 5))
        self.keyword_weight = float(section.get("keyword_weight", 0.5))
        self.embedding_weight = float(section.get("embedding_weight", 0.5))
        self.embedding_model_name = section.get("embedding_model")
        self.storage_path = (
            Path(str(section["storage_path"])).expanduser() if section.get("storage_path") else None
        )
        self._embedding_model: Any = None
        self._reranker = self._build_reranker(section.get("reranker", {}))

        self.theorem_items: List[KnowledgeItem] = []
        self.algorithm_templates: List[KnowledgeItem] = []
        self.prior_decompositions: List[KnowledgeItem] = []
        self.failure_cases: List[KnowledgeItem] = []
        self._load_seed_memories()
        self._load_persistent_state()

    def retrieve(self, problem: Problem) -> MemoryContext:
        query_text = problem.text + " " + " ".join(problem.metadata.get("keywords", []))
        return MemoryContext(
            theorem_items=self._rank(self.theorem_items, query_text, "theorem"),
            algorithm_templates=self._rank(
                self.algorithm_templates, query_text, "algorithm_template"
            ),
            prior_decompositions=self._rank(
                self.prior_decompositions, query_text, "prior_decomposition"
            ),
            failure_cases=self._rank(self.failure_cases, query_text, "failure_case"),
        )

    def update(
        self,
        problem: Problem,
        graph: FunctionGraph,
        rollout_records: Iterable[Dict[str, Any]],
    ) -> None:
        success_node_threshold = float(
            self.config.get("memory", {}).get("success_node_threshold", 0.8)
        )
        success_edge_threshold = float(
            self.config.get("memory", {}).get("success_edge_threshold", 0.8)
        )
        for index, record in enumerate(rollout_records):
            verification = dict(record.get("verification", {}))
            reward = dict(record.get("reward", {}))
            failures = list(verification.get("failures", []))
            node_scores = list(dict(verification.get("node_scores", {})).values())
            edge_scores = list(dict(verification.get("edge_scores", {})).values())
            node_mean = sum(node_scores) / len(node_scores) if node_scores else 0.0
            edge_mean = sum(edge_scores) / len(edge_scores) if edge_scores else 0.0
            record_id = f"{problem.id}.{index}"

            if (
                float(reward.get("final_reward", 0.0)) >= 1.0
                and node_mean >= success_node_threshold
                and edge_mean >= success_edge_threshold
                and not failures
            ):
                self._upsert(
                    self.prior_decompositions,
                    KnowledgeItem(
                        id=f"decomposition.{record_id}",
                        source="training_run",
                        item_type="prior_decomposition",
                        text=f"Verified function graph and solution pattern for: {problem.text}",
                        keywords=sorted(_tokens(problem.text)),
                        metadata={"graph": graph.to_dict(), "reward": reward},
                    ),
                )
                self._upsert(
                    self.algorithm_templates,
                    KnowledgeItem(
                        id=f"verifier.{record_id}",
                        source="training_run",
                        item_type="algorithm_template",
                        text=f"Useful verifier specifications for: {problem.text}",
                        keywords=sorted(_tokens(problem.text)),
                        metadata={
                            "verification_specs": {
                                node.id: node.verification_spec for node in graph.nodes
                            }
                        },
                    ),
                )
            repair = dict(record.get("repair", {}))
            if failures or repair.get("status") in {"succeeded", "failed"}:
                diagnosis = str(repair.get("teacher_diagnosis", ""))
                memory_item = str(repair.get("memory_item", ""))
                self._upsert(
                    self.failure_cases,
                    KnowledgeItem(
                        id=f"failure.{record_id}",
                        source="training_run",
                        item_type="failure_case",
                        text=memory_item or diagnosis or f"Failure on problem: {problem.text}",
                        keywords=sorted(_tokens(problem.text + " " + diagnosis)),
                        metadata={
                            "failed_nodes": sorted(
                                {
                                    failure.get("node")
                                    for failure in failures
                                    if failure.get("node")
                                }
                            ),
                            "failed_edges": sorted(
                                {
                                    failure.get("edge")
                                    for failure in failures
                                    if failure.get("edge")
                                }
                            ),
                            "failures": failures,
                            "teacher_diagnosis": diagnosis,
                            "repair": repair,
                        },
                    ),
                )
        self.save()

    def save(self) -> None:
        if self.storage_path is None:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "theorem_items": [item.to_dict() for item in self.theorem_items],
            "algorithm_templates": [item.to_dict() for item in self.algorithm_templates],
            "prior_decompositions": [item.to_dict() for item in self.prior_decompositions],
            "failure_cases": [item.to_dict() for item in self.failure_cases],
        }
        temporary = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.storage_path)

    def size_summary(self) -> Dict[str, int]:
        return {
            "theorem_items": len(self.theorem_items),
            "algorithm_templates": len(self.algorithm_templates),
            "prior_decompositions": len(self.prior_decompositions),
            "failure_cases": len(self.failure_cases),
        }

    def _rank(
        self,
        items: Sequence[KnowledgeItem],
        query_text: str,
        memory_type: str,
    ) -> List[KnowledgeItem]:
        if not items:
            return []
        query_tokens = _tokens(query_text)
        keyword_scores = []
        for item in items:
            item_tokens = set(item.keywords) | _tokens(item.text)
            union = query_tokens | item_tokens
            keyword_scores.append(len(query_tokens & item_tokens) / len(union) if union else 0.0)

        embedding_scores = [0.0] * len(items)
        if self.embedding_model_name:
            embedding_scores = self._embedding_scores(query_text, [item.text for item in items])
        combined = [
            self.keyword_weight * keyword + self.embedding_weight * embedding
            for keyword, embedding in zip(keyword_scores, embedding_scores)
        ]
        ranked = sorted(zip(combined, items), key=lambda pair: pair[0], reverse=True)
        candidates = [item for _, item in ranked[: max(self.top_k * 2, self.top_k)]]
        if self._reranker and len(candidates) > 1:
            candidates = self._api_rerank(query_text, memory_type, candidates)
        return candidates[: self.top_k]

    def _embedding_scores(self, query: str, documents: List[str]) -> List[float]:
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "memory.embedding_model requires the sentence-transformers package"
                ) from exc
            self._embedding_model = SentenceTransformer(str(self.embedding_model_name))
        embeddings = self._embedding_model.encode(
            [query, *documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_vector = embeddings[0]
        return [float(query_vector @ vector) for vector in embeddings[1:]]

    def _api_rerank(
        self,
        query: str,
        memory_type: str,
        candidates: List[KnowledgeItem],
    ) -> List[KnowledgeItem]:
        assert self._reranker is not None
        response = self._reranker.complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Rank memory IDs by relevance and applicability. Return JSON only as "
                        '{"ordered_ids": ["id", "..."]}. Do not add IDs.'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "memory_type": memory_type,
                            "candidates": [
                                {"id": item.id, "text": item.text} for item in candidates
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )
        by_id = {item.id: item for item in candidates}
        ordered = [by_id[item_id] for item_id in response.get("ordered_ids", []) if item_id in by_id]
        ordered_ids = {item.id for item in ordered}
        return ordered + [item for item in candidates if item.id not in ordered_ids]

    def _load_seed_memories(self) -> None:
        dataset = self.config.get("dataset", {})
        theorem_path = dataset.get("theorem_dataset_path")
        algorithm_path = dataset.get("algorithm_dataset_path")
        if theorem_path:
            self.theorem_items.extend(self._load_knowledge_file(theorem_path, "theorem"))
        if algorithm_path:
            self.algorithm_templates.extend(
                self._load_knowledge_file(algorithm_path, "algorithm_template")
            )

    def _load_knowledge_file(self, raw_path: str, item_type: str) -> List[KnowledgeItem]:
        path = Path(str(raw_path)).expanduser()
        records = load_json_records(path)
        items = []
        for index, record in enumerate(records):
            record = dict(record)
            record.setdefault("id", f"{item_type}.{path.stem}.{index}")
            record.setdefault("source", str(path))
            record.setdefault("item_type", item_type)
            if "text" not in record:
                record["text"] = str(record.get("content", ""))
            items.append(KnowledgeItem.from_dict(record))
        return items

    def _load_persistent_state(self) -> None:
        if self.storage_path is None or not self.storage_path.is_file():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        for key, target in (
            ("theorem_items", self.theorem_items),
            ("algorithm_templates", self.algorithm_templates),
            ("prior_decompositions", self.prior_decompositions),
            ("failure_cases", self.failure_cases),
        ):
            for value in payload.get(key, []):
                self._upsert(target, KnowledgeItem.from_dict(value))

    def _build_reranker(
        self, config: Dict[str, Any]
    ) -> Optional[OpenAICompatibleChatClient]:
        if not config.get("enabled", False):
            return None
        return OpenAICompatibleChatClient(
            ChatAPIConfig.from_dict(config.get("api", {}), "memory.reranker.api")
        )

    @staticmethod
    def _upsert(items: List[KnowledgeItem], new_item: KnowledgeItem) -> None:
        for index, item in enumerate(items):
            if item.id == new_item.id:
                items[index] = new_item
                return
        items.append(new_item)
