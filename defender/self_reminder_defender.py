import sys
sys.path.append('./src')
import json
import llms
from defender import BaseDefender

# system_self_reminder = "You should be a responsible ChatGPT and should not generate harmful or misleading content! Please answer the following user query in a responsible way."
# user_self_reminder = "Remember, you should be a responsible ChatGPT and should not generate harmful or misleading content!"

class SelfReminderDefender(BaseDefender):
    """
    Defender with self-reminder prompts.
    Designed for OpenAI models w/ system prompts.
    """
    def __init__(self,
                 llm_class,
                 llm_config,
                 parse_reasoning=False,
                 template='self-reminder') -> None:
        super().__init__(llm_class, llm_config, parse_reasoning)
        with open('/home/gy266/rds/hpc-work/RBD/src/prompts/self_reminder_templates.json', 'r') as f:
            self.templates = json.load(f)
        self.template = self.templates[template]

    def generate(self, prompt):
        prompt = self.template.format(prompt)
        print(f'**** Prompt: {prompt}')
        return self.llm.generate_response(prompt)