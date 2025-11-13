import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import os
import json

class CoTEvaluator:
    def __init__(
        self,
        model_name: str,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        batch_size: int = 4,
        api_key: str = None,
        use_vllm: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.use_vllm = use_vllm

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )

        if use_vllm:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available! vLLM requires GPU.")
            self.model = LLM(
                model=model_name,
                dtype="float16",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
            )

            self.sampling_params = SamplingParams(
                                    max_tokens=self.max_tokens,
                                    temperature=self.temperature,
                                    top_p=1.0,
                                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
    
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    
    def create_prompt(self, context: str, question: str, options: List[str], language: str) -> str:
        instructions = {
            "en": "Think through this question step by step. Explain your reasoning, then provide your final answer as A, B, or C.",
            "es": "Piensa en esta pregunta paso a paso. Explica tu razonamiento y luego proporciona tu respuesta final como A, B o C.",
            "tr": "Bu soruyu adım adım düşünün. Mantığınızı açıklayın, sonra nihai cevabınızı A, B veya C olarak verin.",
            "nl": "Denk stap voor stap na over deze vraag. Leg uw redenering uit en geef dan uw definitieve antwoord als A, B of C.",
        }
        
        instruction = instructions.get(language, instructions["en"])
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        
        prompt = f"""Context: {context}
        
                    Question: {question}

                    Options: {options_text}

                    {instruction}

                    Step-by-step reasoning:"""
        return prompt
    
    def generate(self, prompts: List[str]) -> List[str]:
        generated_texts = []
        with torch.no_grad():
            if self.use_vllm:
                outputs = self.model.generate(prompts, self.sampling_params)
                for output in outputs:   
                    completion = output.outputs[0]  # CompletionOutput
                    text = completion.text
                    generated_texts.append(text.strip()) 
            else:
                inputs = self.tokenizer(
                                    prompts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=2048
                                ).to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                for i, output in enumerate(outputs):
                    input_length = inputs["input_ids"][i].shape[0]
                    generated = output[input_length:]
                    text = self.tokenizer.decode(generated, skip_special_tokens=True)
                    generated_texts.append(text)
        return generated_texts

    def extract_answer(self, generated_text: str) -> Tuple[str, str]:
        text = generated_text.strip()
        
        markers = ["Final Answer:", "Answer:", "final answer:", "answer:", "Therefore,"]
        reasoning_part = text
        answer_part = ""
        
        for marker in markers:
            if marker.lower() in text.lower():
                idx = text.lower().index(marker.lower())
                reasoning_part = text[:idx].strip()
                answer_part = text[idx:].strip()
                break
        
        answer = "UNKNOWN"
        search_text = answer_part if answer_part else text[-100:]
        
        for char in search_text:
            if char.upper() in ["A", "B", "C"]:
                answer = char.upper()
                break
        
        return reasoning_part, answer