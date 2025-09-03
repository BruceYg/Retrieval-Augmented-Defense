import sys
sys.path.append('./src')
import llms
from pathlib import Path
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from defender import BaseDefender

class LlamaGuardDefender(BaseDefender):
    def __init__(self,
                 llm_class,
                 llm_config,
                 mod_model,
                 parse_reasoning=False,
                 double_check=False,
                 ) -> None:
        super().__init__(llm_class, llm_config)
        self.llm = getattr(llms, llm_class)(**llm_config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model = AutoModelForCausalLM.from_pretrained(mod_model, torch_dtype=torch.bfloat16, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(mod_model)
        self.parse_reasoning = parse_reasoning
        self.double_check = double_check
        self.model.eval()

    def generate(self, prompt):
        conversation = [
            {
                "role": "user",
                "content": prompt
            },
        ]

        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=20,
                pad_token_id=0,
                top_p=1.0,
                temperature=1.0,
                do_sample=False,
            )
        generated_tokens = output[:, prompt_len:]
        generated = self.tokenizer.decode(generated_tokens[0])
        print(f'********** {generated} **********')
        
        if 'safe' in generated and 'unsafe' not in generated:
            model_response = self.llm.generate_response(prompt)
            if not self.double_check:
                return model_response
        elif 'unsafe' in generated:
            return "I'm sorry, but I can't help with that."
        else:
            model_response = self.llm.generate_response(prompt)
            if not self.double_check:
                return model_response
        conversation = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": model_response,
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt").to(self.model.device)
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=20,
                pad_token_id=0,
                top_p=1.0,
                temperature=1.0,
                do_sample=False,
            )
        generated_tokens = output[:, prompt_len:]
        generated = self.tokenizer.decode(generated_tokens[0])
        print(f'**********Second round: {generated} **********')
        if 'safe' in generated and 'unsafe' not in generated:
            return model_response
        elif 'unsafe' in generated:
            return "I'm sorry, but I can't help with that."
        else:
            return model_response