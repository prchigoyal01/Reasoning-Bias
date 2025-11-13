import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import os
import json


class ShortAnswerEvaluator:
    def __init__(
        self,
        model_name: str,
        temperature: float = 1.0,
        max_tokens: int = 16,
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
                        dtype="float16",            # or "fp16" if bf16 not supported
                        # maybe specify device_map, quantization, etc
                        # e.g.: load_in_4bit=True if supported
                        # additional engine kwargs: max_batch_size, num_workers, etc
                        tensor_parallel_size=1,  # Add this: use 1 GPU
                        gpu_memory_utilization=0.9,  # Optional: use 90% of GPU memory
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
            "en": "Consider the given description and choose the appropriate answer to the question by selecting one option among A, B, or C. Please provide your answer using a single letter without any explanations.",
            "es": "Considere la descripción dada y elija la respuesta adecuada a la pregunta seleccionando una opción entre A, B o C. Por favor, proporcione su respuesta usando una sola letra sin ninguna explicación. Descripción: {} Pregunta: {} A: {} B: {} C: {} Respuesta:",
            "tr": "Verilen açıklamayı göz önünde bulundurarak soruya uygun cevabı A, B veya C seçeneğini seçerek verin. Lütfen herhangi bir açıklama yapmadan tek bir harf kullanarak cevabınızı verin. Açıklama: {} Soru: {} A: {} B: {} C: {} Cevap:",
            "nl": "Overweeg de gegeven beschrijving en kies het juiste antwoord op de vraag door één optie te selecteren tussen A, B of C. Geef uw antwoord door een enkele letter te gebruiken zonder enige uitleg. Beschrijving: {} Vraag: {} A: {} B: {} C: {} Antwoord:",
        }
        
        instruction = instructions.get(language, instructions["en"])
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        
        prompt = f"""Context: {context}
        
                    Question: {question}

                    Options: {options_text}

                    {instruction}

                    Answer:"""
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
    
    def extract_answer(self, generated_text: str) -> str:
        text = generated_text.strip()
        
        markers = ["Final Answer:", "Answer:", "final answer:", "answer:"]
        search_text = text
        
        for marker in markers:
            if marker in text:
                parts = text.split(marker, 1)
                search_text = parts[1].strip() if len(parts) > 1 else text
                break
        
        # Extract answer choice - check first 50 chars first
        for char in search_text[:50]:
            if char.upper() in ["A", "B", "C"]:
                return "", char.upper()
        
        # If not found in first 50 chars, search entire text
        for char in ["A", "B", "C"]:
            if char in search_text.upper():
                return "", char
        
        return "", "UNKNOWN"