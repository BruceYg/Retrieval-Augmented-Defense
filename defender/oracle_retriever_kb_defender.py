import sys
sys.path.append('./src')
import llms
from defender import BaseDefender


class OracleRetrieverKBDefender(BaseDefender):
    """
    Defender with oracle retriever and oracle KB.
    """
    def __init__(self,
                 llm_class,
                 llm_config,
                 prompt_file) -> None:
        super().__init__(llm_class, llm_config)
        with open(prompt_file, 'r') as f:
            self.prompt_format = f.read()
        print('Double oracle defender initialized')
    
    def generate(self, prompt, harmful_example=None):
        retrieved_question = prompt if harmful_example is None else harmful_example
        placeholders = {'harmful question': retrieved_question, 'user query': prompt}
        final_prompt = self.prompt_format.format(**placeholders)
        return self.llm.generate_response(final_prompt)