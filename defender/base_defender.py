import sys
sys.path.append('./src')
import llms

class BaseDefender:
    """
    Base defender with LLM only.
    """
    def __init__(self,
                 llm_class,
                 llm_config,
                 parse_reasoning=False) -> None:
        self.llm = getattr(llms, llm_class)(**llm_config)
        self.parse_reasoning = parse_reasoning

    def generate(self, prompt):
        return self.llm.generate_response(prompt)

    def delete(self):
        self.llm.delete()