# Adapted from https://github.com/IAAR-Shanghai/xVerify/blob/main/src/xVerify/model.py

from typing import Tuple, List
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from .prompts import BASE_TEMPLATE
import torch
import gc


class Model:
    """Local xVerify model for verification tasks. Processes prompts in batches."""

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.template = self._load_template()
        self.tokenizer, self.model = self._initialize_local_model()
        self.model.eval()

    def _load_template(self) -> str:
        """Loads the prompt template for the specified model."""
        try:
            return BASE_TEMPLATE[self.model_name.split("/")[-1]]
        except KeyError:
            logger.error(f"Base template for model '{self.model_name}' not found.")
            raise
        except Exception:
            logger.exception("An unexpected error occurred while loading the template.")
            raise

    def _initialize_local_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Initializes and loads the local model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto", 
            device_map="auto",
            trust_remote_code=True
        )
        return tokenizer, model

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generates responses for a batch of prompts."""

        base_template = self._load_template()
        formatted_prompts = [base_template.format(query=prompt) for prompt in prompts]
        
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 
        )
        
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"].to(self.model.device),
                attention_mask=inputs["attention_mask"].to(self.model.device),
                max_new_tokens=self.max_tokens,
                do_sample=False,
                use_cache=True 
            )
        
        responses = self.tokenizer.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        del inputs, output_ids, formatted_prompts
        torch.cuda.empty_cache(); gc.collect()
        
        return [response.strip() for response in responses]