from adapters import create_llm_adapter
import yaml

class Agent:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.llm = create_llm_adapter(config.get("llm", {}))

    def process(self, user_input: str) -> str:
        return self.llm.generate(user_input)
