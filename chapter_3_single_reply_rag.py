import faiss
import numpy as np
import json
import ollama
from sentence_transformers import SentenceTransformer
from configs.settings import BaseSettings, OllamaSettings

# 檢索 FAISS，取得相似文本
def retrieve_similar_texts(query, k=3):
    """
    從 FAISS 取得最相似的 k 筆文本

    參數:
        query: 查詢文本
        k: 返回的文本數量

    返回:
        list: 最相似的 k 筆文本
    """

    # 讀取 JSON 原始文本
    json_file = BaseSettings.output_file_path
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 保留文本以便檢索
    texts = [item["page_content"] for item in data] 

    # 載入 E5 向量模型
    model = SentenceTransformer(BaseSettings.vector_model)

    # 轉換查詢文本為向量
    query_vector = model.encode([query], normalize_embeddings=True)
    
    # 轉換為 numpy array
    query_vector_np = np.array(query_vector, dtype=np.float32)

    # 讀取 FAISS 索引
    faiss_index = faiss.read_index(BaseSettings.vector_index_path)
    
    # 檢索最相似的 k 筆文本
    distances, indices = faiss_index.search(query_vector_np, k)

    retrieved_texts = [texts[idx] for idx in indices[0]]
    return retrieved_texts

# 透過 AI 產生回答
def generate_answer(query):
    """
    使用 FAISS + AI 產生 RAG 回應

    參數:
        query: 查詢文本

    返回:
        str: RAG 回應
    """
    retrieved_texts = retrieve_similar_texts(query)

    # 組合查詢內容 & FAISS 檢索內容
    context = "\n\n".join(retrieved_texts)
    prompt = f"根據以下資料回答問題：\n\n{context}\n\n問題：{query}\n\n請提供詳細回答。"

    # 呼叫 Ollama
    response = ollama.chat(model=OllamaSettings.model_llama_3_1, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

# 執行
def run_process():
    query_text = "請說明軟體需求規格"
    answer = generate_answer(query_text)

    print("=== AI 回答 ===")
    print(answer)

if __name__ == "__main__":
    run_process()