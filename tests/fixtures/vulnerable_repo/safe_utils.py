def add(a: int, b: int) -> int:
    return a + b


def sanitize_input(text: str) -> str:
    return text.replace("'", "''").replace(";", "")
