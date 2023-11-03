import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def calculate_price(input_tokens: int, output_tokens: int) -> float:
    input_cost = input_tokens * (0.0015 / 1000)
    output_cost = output_tokens * (0.002 / 1000)
    total_cost = input_cost + output_cost
    return total_cost