import sys
sys.path.append('./src')
import llms
from pathlib import Path
import numpy as np
import torch
from defender import BaseDefender

class IADefender(BaseDefender):
    def __init__(self,
                 llm_class,
                 llm_config,
                 parse_reasoning=False,
                 ) -> None:
        super().__init__(llm_class, llm_config)
        self.llm = getattr(llms, llm_class)(**llm_config)
        self.parse_reasoning = parse_reasoning
        self.ia_prompt = "Please identify the essential intention behind the following user query with a focus on safety, ethics, and legality. Ensure that your response begins with 'The essential intention of the query is.' Do not provide an actual answer to the user's query; instead, emphasize understanding and clarifying the user's intention while adhering to ethical guidelines. Following is the query you should focus on:\n"
        self.ct_prompt = "Knowing its essential intention, now directly answer the aforementioned query with your normal response. Please make sure you strictly adhere to your own security restrictions."

    def parser(self, response):
        reasoning, response = response.split('</think>')[0].strip(), response.split('</think>')[-1].strip()
        if 'Answer:' in response:
            post_reasoning = response.split('Answer:')[0].strip()
            reasoning = f"{reasoning}\n{post_reasoning}"
            response = response.split('Answer:')[1].strip()
            response = response[2:].strip() if response[0:2] == '**' else response
        return reasoning, response
    
    def generate(self, prompt):
        ia_prompt = f"{self.ia_prompt}'''\n{prompt}\n'''"
        ia_response = self.llm.generate_response(ia_prompt)
        if self.parse_reasoning:
            _, ia_response = self.parser(ia_response)
        print(f'****ia_response: {ia_response}')
        conversation = [ia_prompt, ia_response, self.ct_prompt]
        response = self.llm.generate_response_multi_turn(conversation)
        print(f'****response: {response}')
        print('-'*100)
        return response