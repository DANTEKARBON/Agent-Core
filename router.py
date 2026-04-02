import yaml
import re
from adapters import create_llm_adapter

def classify_query(query, config_path="config.yaml"):
    """Определяет, какую модель использовать для запроса."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    router_config = config["router"]
    model_name = router_config["model_name"]
    prompt_template = router_config["prompt_template"]
    
    model_config = config["models"][model_name]
    adapter = create_llm_adapter(model_config)
    
    prompt = prompt_template.format(query=query)
    response = adapter.generate(prompt)
    
    # Очистка ответа: убираем специальные токены, оставляем только первое слово
    cleaned = re.sub(r'<\|.*?\|>', '', response)   # удаляем <|end|>, <|user|> и т.п.
    words = cleaned.strip().split()
    decision = words[0].lower() if words else "phi3"
    
    # Если ответ не один из ожидаемых, используем phi3 как запасной
    if decision not in ["coder", "reasoner"]:
        decision = "phi3"
    
    return decision
