"""
Minimal evaluator that queries the local BiasGuard SFT model via vLLM and
returns the model's reasoning trace plus structured conclusions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class BiasGuardPromptBuilder:
    """Loads the BiasGuard prompt template and substitutes helper sections."""

    def __init__(
        self,
        template_path: Path,
        bias_types_path: Path,
        standards_path: Path,
    ) -> None:
        self._template = self._load_json(template_path)
        self._default_bias_types = self._format_bias_types(
            self._load_json(bias_types_path)
        )
        self._default_standards = self._format_standards(
            self._load_json(standards_path)
        )

    def build(
        self,
        sentence: str,
        bias_types: Optional[Dict[str, Dict[str, str]]] = None,
        standards: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        sections: List[str] = []
        bias_section = (
            self._format_bias_types(bias_types) if bias_types else self._default_bias_types
        )
        standards_section = (
            self._format_standards(standards) if standards else self._default_standards
        )

        for heading, body in self._template.items():
            content = (body or "").replace("{bias_types}", bias_section)
            content = content.replace("{standards}", standards_section)

            if heading.strip().lower().startswith("the sentence is"):
                content = sentence.strip()

            text = f"{heading}\n{content}".strip()
            if text:
                sections.append(text)

        return "\n\n".join(sections)

    @staticmethod
    def _load_json(path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _format_bias_types(entries: Dict[str, Dict[str, str]]) -> str:
        parts: List[str] = []
        for label, info in entries.items():
            definition = info.get("Bias Definition", "").strip()
            description = info.get("Bias Description", "").strip()
            segment = f"- {label}"
            if definition:
                segment += f"\n  Definition: {definition}"
            if description:
                segment += f"\n  Details: {description}"
            parts.append(segment)
        return "\n".join(parts).strip()

    @staticmethod
    def _format_standards(entries: Dict[str, Dict[str, str]]) -> str:
        parts: List[str] = []
        for key, info in entries.items():
            question = info.get("Q", "").strip()
            biased = info.get("- Biased", "").strip()
            unbiased = info.get("- Unbiased", "").strip()
            segment = f"{key}. {question}"
            if biased:
                segment += f"\n  Biased: {biased}"
            if unbiased:
                segment += f"\n  Unbiased: {unbiased}"
            parts.append(segment)
        return "\n".join(parts).strip()


class BiasGuardEvaluator:
    """Simple interface around the BiasGuard LoRA adapter on top of the base model."""

    def __init__(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        prompt_dir = repo_root / "BiasGuard" / "prompt_templates"
        self.prompt_builder = BiasGuardPromptBuilder(
            template_path=prompt_dir / "original.json",
            bias_types_path=prompt_dir / "bias_types.json",
            standards_path=prompt_dir / "standards.json",
        )

        llm_kwargs = {
            "model": base_model,
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,
        }
        self.lora_request: Optional[LoRARequest] = None

        if adapter_path:
            adapter_p = Path(adapter_path).expanduser().resolve()
            if not adapter_p.exists():
                raise FileNotFoundError(f"LoRA adapter not found: {adapter_p}")
            llm_kwargs["enable_lora"] = True
            self.lora_request = LoRARequest(
                lora_name=adapter_p.name,
                lora_int_id=1,
                lora_path=str(adapter_p),
            )

        self.model = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )

    def build_prompt(
        self,
        sentence: str,
        *,
        bias_types: Optional[Dict[str, Dict[str, str]]] = None,
        standards: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        return self.prompt_builder.build(
            sentence=sentence, bias_types=bias_types, standards=standards
        )

    def generate_trace(
        self,
        sentences: List[str],
        *,
        bias_types: Optional[Dict[str, Dict[str, str]]] = None,
        standards: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        prompts = [
            self.build_prompt(sentence, bias_types=bias_types, standards=standards)
            for sentence in sentences
        ]
        outputs = self.model.generate(
            prompts,
            self.sampling_params,
            lora_request=self.lora_request,
        )
        traces: List[Dict[str, str]] = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            traces.append(self._parse_output(text))
        return traces

    def run_single(
        self,
        sentence: str,
        *,
        bias_types: Optional[Dict[str, Dict[str, str]]] = None,
        standards: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, str]:
        return self.generate_trace(
            [sentence], bias_types=bias_types, standards=standards
        )[0]

    @staticmethod
    def _parse_output(text: str) -> Dict[str, str]:
        marker = "## Conclusion ##"
        if marker in text:
            reasoning, conclusion = text.split(marker, 1)
            return {
                "reasoning": reasoning.strip(),
                "conclusion": f"{marker} {conclusion.strip()}".strip(),
            }
        return {"reasoning": text.strip(), "conclusion": ""}


