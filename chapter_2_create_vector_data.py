import json
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from configs.settings import BaseSettings

# 讀取 JSON 文件
def load_json(file_path):
    """
    從指定路徑載入 JSON 文件
    
    參數:
        file_path: JSON 檔案路徑
    
    返回:
        dict: 載入的 JSON 資料
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 建立 FAISS 索引
def create_faiss_index(data):
    """
    使用 FAISS 建立索引
    
    參數:
        data: JSON 資料
    
    返回:
        faiss.Index: FAISS 索引物件
    """

    # 提取所有文本段落
    texts = [item["page_content"] for item in data if "page_content" in item]

    # 載入 E5 嵌入模型
    model = SentenceTransformer(BaseSettings.vector_model)

    # 產生向量
    embeddings = model.encode(texts, normalize_embeddings=True)
        
    # 取得向量維度
    embedding_dim = embeddings.shape[1]
    
    # 使用 L2 距離建立索引
    index = faiss.IndexFlatL2(embedding_dim)
    
    # 確保向量是 float32 格式
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
        
    # 加入向量
    index.add(embeddings)
    
    return index

# 儲存 FAISS 索引
def save_faiss_index(index, file_path):
    """
    儲存 FAISS 索引
    
    參數:
        index: FAISS 索引
        file_path: 檔案路徑
    """
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 儲存檔案
    faiss.write_index(index, file_path)
    
    print(f"向量索引已儲存至: {file_path}")

# 執行
def run_process():
    # Step 1: 載入 JSON 文件
    data = load_json(BaseSettings.output_file_path)

    # Step 2: 建立 FAISS 索引
    faiss_index = create_faiss_index(data)

    # Step 3: 儲存索引
    save_faiss_index(faiss_index, BaseSettings.vector_index_path)

if __name__ == "__main__":
    run_process()