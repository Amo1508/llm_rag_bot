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
    need_to_parse_file_path = f"data/raw_data/temple_doc.pdf"
    # 輸出資料夾路徑
    output_file_path = f"data/processed_data/temple_doc.json"
    # 向量轉化模型
    vector_model = "intfloat/multilingual-e5-large"
    # 向量索引路徑_單向
    vector_index_single_path = f"data/processed_data/"
    # 向量索引文件_單向
    vector_index_single_file = "temple_doc.faiss"
    # 向量索引路徑_雙向
    vector_index_path = f"data/processed_data"
