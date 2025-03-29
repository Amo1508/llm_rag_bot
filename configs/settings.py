class OllamaSettings:
    """
    Ollama 設定
    """
    # Ollama 模型 3.1 8B
    model_llama_3_1 = "llama3.1:latest"
    # Ollama 服務 URL
    service_url = "http://localhost:11434"
    
class BaseSettings:
    """
    基礎設定
    """
    # 需解析的資料夾路徑
    need_to_parse_file_path = "data/raw_data/開發規範.pdf"
    # 輸出資料夾路徑
    output_file_path = "data/processed_data/開發規範.json"
