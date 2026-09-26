"""Shared data structures for the FSG-RL pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]


def _list_from_dict(cls, values: Optional[List[JsonDict]]) -> List[Any]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{cls.__name__} collection must be a JSON array")
    result = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(
                f"{cls.__name__} item at index {index} must be a JSON object, "
                f"got {type(item).__name__}"
            )
        result.append(cls.from_dict(item))
    return result


@dataclass
class Problem:
    id: str
    text: str
    gold_answer: Optional[str] = None
    split: str = "train"
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "Problem":
        return cls(
            id=data["id"],
            text=data["text"],
            gold_answer=data.get("gold_answer"),
            split=data.get("split", "train"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class KnowledgeItem:
    id: str
    source: str
    item_type: str
    text: str
    keywords: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "KnowledgeItem":
        return cls(
            id=data["id"],
            source=data.get("source", ""),
            item_type=data.get("item_type", ""),
            text=data.get("text", ""),
            keywords=list(data.get("keywords", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MemoryContext:
    theorem_items: List[KnowledgeItem] = field(default_factory=list)
    algorithm_templates: List[KnowledgeItem] = field(default_factory=list)
    prior_decompositions: List[KnowledgeItem] = field(default_factory=list)
    failure_cases: List[KnowledgeItem] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "theorem_items": [item.to_dict() for item in self.theorem_items],
            "algorithm_templates": [item.to_dict() for item in self.algorithm_templates],
            "prior_decompositions": [item.to_dict() for item in self.prior_decompositions],
            "failure_cases": [item.to_dict() for item in self.failure_cases],
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "MemoryContext":
        return cls(
            theorem_items=_list_from_dict(KnowledgeItem, data.get("theorem_items")),
            algorithm_templates=_list_from_dict(KnowledgeItem, data.get("algorithm_templates")),
            prior_decompositions=_list_from_dict(KnowledgeItem, data.get("prior_decompositions")),
            failure_cases=_list_from_dict(KnowledgeItem, data.get("failure_cases")),
        )


@dataclass
class FunctionNode:
    id: str
    name: str
    question: str
    signature: str
    expected_output_type: str
    verification_spec: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "FunctionNode":
        verification_spec = data.get("verification_spec", {})
        if not isinstance(verification_spec, dict):
            raise ValueError(
                f"Node {data.get('id', '<unknown>')!r} verification_spec must be "
                f"a JSON object, got {type(verification_spec).__name__}"
            )
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            question=data.get("question", ""),
            signature=data.get("signature", ""),
            expected_output_type=data.get("expected_output_type", ""),
            verification_spec=dict(verification_spec),
        )


@dataclass
class DependencyEdge:
    source: str
    target: str
    relation_type: str
    check_method: str
    severity: str = "normal"

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "DependencyEdge":
        check_method = data.get("check_method", "")
        if isinstance(check_method, dict):
            check_method = json.dumps(
                check_method, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        elif not isinstance(check_method, str):
            raise ValueError(
                f"Edge {data.get('source', '<unknown>')!r}->"
                f"{data.get('target', '<unknown>')!r} check_method must be a JSON "
                f"string or object, got {type(check_method).__name__}"
            )
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=data.get("relation_type", ""),
            check_method=check_method,
            severity=data.get("severity", "normal"),
        )


@dataclass
class FunctionGraph:
    problem_id: str
    nodes: List[FunctionNode]
    edges: List[DependencyEdge]

    def to_dict(self) -> JsonDict:
        return {
            "problem_id": self.problem_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "FunctionGraph":
        return cls(
            problem_id=data["problem_id"],
            nodes=_list_from_dict(FunctionNode, data.get("nodes")),
            edges=_list_from_dict(DependencyEdge, data.get("edges")),
        )

    def node_by_id(self, node_id: str) -> Optional[FunctionNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None


@dataclass
class ParsedFunctionSpan:
    node_id: str
    raw_text: str
    code_blocks: List[str] = field(default_factory=list)
    extracted_answer: Optional[str] = None
    extracted_formula: Optional[str] = None
    extracted_values: JsonDict = field(default_factory=dict)
    start_char: int = -1
    end_char: int = -1

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ParsedFunctionSpan":
        return cls(
            node_id=data["node_id"],
            raw_text=data.get("raw_text", ""),
            code_blocks=list(data.get("code_blocks", [])),
            extracted_answer=data.get("extracted_answer"),
            extracted_formula=data.get("extracted_formula"),
            extracted_values=dict(data.get("extracted_values", {})),
            start_char=int(data.get("start_char", -1)),
            end_char=int(data.get("end_char", -1)),
        )


@dataclass
class PolicyRollout:
    problem_id: str
    raw_text: str
    parsed_spans: List[ParsedFunctionSpan] = field(default_factory=list)
    prompt_text: str = ""
    backend: str = ""
    generation_seconds: float = 0.0
    repaired: bool = False
    prompt_token_ids: List[int] = field(default_factory=list, repr=False)
    completion_token_ids: List[int] = field(default_factory=list, repr=False)
    span_token_ranges: Dict[str, List[int]] = field(default_factory=dict, repr=False)

    def to_dict(self) -> JsonDict:
        return {
            "problem_id": self.problem_id,
            "raw_text": self.raw_text,
            "parsed_spans": [span.to_dict() for span in self.parsed_spans],
            "prompt_text": self.prompt_text,
            "backend": self.backend,
            "generation_seconds": self.generation_seconds,
            "repaired": self.repaired,
            "prompt_token_count": len(self.prompt_token_ids),
            "completion_token_count": len(self.completion_token_ids),
            "span_token_ranges": dict(self.span_token_ranges),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "PolicyRollout":
        return cls(
            problem_id=data["problem_id"],
            raw_text=data.get("raw_text", ""),
            parsed_spans=_list_from_dict(ParsedFunctionSpan, data.get("parsed_spans")),
            prompt_text=data.get("prompt_text", ""),
            backend=data.get("backend", ""),
            generation_seconds=float(data.get("generation_seconds", 0.0)),
            repaired=bool(data.get("repaired", False)),
            prompt_token_ids=[int(value) for value in data.get("prompt_token_ids", [])],
            completion_token_ids=[int(value) for value in data.get("completion_token_ids", [])],
            span_token_ranges={
                str(key): [int(value) for value in values]
                for key, values in dict(data.get("span_token_ranges", {})).items()
            },
        )


@dataclass
class NodeExecutionResult:
    node_id: str
    executable: bool = False
    stdout: str = ""
    stderr: str = ""
    runtime_seconds: float = 0.0
    timeout: bool = False
    outputs: JsonDict = field(default_factory=dict)
    test_results: List[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "NodeExecutionResult":
        return cls(
            node_id=data["node_id"],
            executable=bool(data.get("executable", False)),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            runtime_seconds=float(data.get("runtime_seconds", 0.0)),
            timeout=bool(data.get("timeout", False)),
            outputs=dict(data.get("outputs", {})),
            test_results=list(data.get("test_results", [])),
        )


@dataclass
class ExecutionResult:
    node_results: Dict[str, NodeExecutionResult] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    runtime_seconds: float = 0.0
    timeout: bool = False
    executable: bool = True

    def to_dict(self) -> JsonDict:
        return {
            "node_results": {
                node_id: result.to_dict() for node_id, result in self.node_results.items()
            },
            "stdout": self.stdout,
            "stderr": self.stderr,
            "runtime_seconds": self.runtime_seconds,
            "timeout": self.timeout,
            "executable": self.executable,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ExecutionResult":
        return cls(
            node_results={
                node_id: NodeExecutionResult.from_dict(result)
                for node_id, result in dict(data.get("node_results", {})).items()
            },
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            runtime_seconds=float(data.get("runtime_seconds", 0.0)),
            timeout=bool(data.get("timeout", False)),
            executable=bool(data.get("executable", True)),
        )


@dataclass
class VerificationResult:
    node_scores: Dict[str, float] = field(default_factory=dict)
    node_execution_scores: Dict[str, float] = field(default_factory=dict)
    node_test_scores: Dict[str, float] = field(default_factory=dict)
    node_property_scores: Dict[str, float] = field(default_factory=dict)
    edge_scores: Dict[str, float] = field(default_factory=dict)
    final_answer_score: float = 0.0
    backward_score: float = 0.0
    consensus_score: float = 0.0
    failures: List[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "VerificationResult":
        return cls(
            node_scores={str(k): float(v) for k, v in dict(data.get("node_scores", {})).items()},
            node_execution_scores={
                str(k): float(v)
                for k, v in dict(data.get("node_execution_scores", {})).items()
            },
            node_test_scores={
                str(k): float(v) for k, v in dict(data.get("node_test_scores", {})).items()
            },
            node_property_scores={
                str(k): float(v)
                for k, v in dict(data.get("node_property_scores", {})).items()
            },
            edge_scores={str(k): float(v) for k, v in dict(data.get("edge_scores", {})).items()},
            final_answer_score=float(data.get("final_answer_score", 0.0)),
            backward_score=float(data.get("backward_score", 0.0)),
            consensus_score=float(data.get("consensus_score", 0.0)),
            failures=list(data.get("failures", [])),
        )

    def has_failures(self) -> bool:
        return bool(self.failures)


@dataclass
class RewardRecord:
    total_reward: float
    node_rewards: Dict[str, float] = field(default_factory=dict)
    edge_rewards: Dict[str, float] = field(default_factory=dict)
    final_reward: float = 0.0
    backward_reward: float = 0.0
    consensus_reward: float = 0.0
    format_gate: float = 0.0
    efficiency_penalty: float = 0.0
    repair_penalty: float = 0.0
    signature_reward: float = 0.0
    execution_reward: float = 0.0
    unit_test_reward: float = 0.0
    property_test_reward: float = 0.0
    teacher_reward: float = 0.0
    answer_gate: float = 1.0
    span_rewards: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "RewardRecord":
        return cls(
            total_reward=float(data.get("total_reward", 0.0)),
            node_rewards={str(k): float(v) for k, v in dict(data.get("node_rewards", {})).items()},
            edge_rewards={str(k): float(v) for k, v in dict(data.get("edge_rewards", {})).items()},
            final_reward=float(data.get("final_reward", 0.0)),
            backward_reward=float(data.get("backward_reward", 0.0)),
            consensus_reward=float(data.get("consensus_reward", 0.0)),
            format_gate=float(data.get("format_gate", 0.0)),
            efficiency_penalty=float(data.get("efficiency_penalty", 0.0)),
            repair_penalty=float(data.get("repair_penalty", 0.0)),
            signature_reward=float(data.get("signature_reward", 0.0)),
            execution_reward=float(data.get("execution_reward", 0.0)),
            unit_test_reward=float(data.get("unit_test_reward", 0.0)),
            property_test_reward=float(data.get("property_test_reward", 0.0)),
            teacher_reward=float(data.get("teacher_reward", 0.0)),
            answer_gate=float(data.get("answer_gate", 1.0)),
            span_rewards={str(k): float(v) for k, v in dict(data.get("span_rewards", {})).items()},
        )
