import json
from configs.settings import BaseSettings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# 此章節需從新建立向量索引庫，與單向索引不同
# 需要重新載入 JSON 文件

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

# 儲存 FAISS 索引
def save_faiss_index(data):
    """
    儲存 FAISS 索引
    
    參數:
        data: JSON 資料
    """
    
    # 轉換 JSON 資料成 LangChain 的 Document 格式
    docs = [Document(page_content=item["page_content"]) for item in data if "page_content" in item]

    # 建立 HuggingFace 嵌入模型
    embeddings = HuggingFaceEmbeddings(model_name=BaseSettings.vector_model)

    # 建立 FAISS 向量資料庫
    vector_db = FAISS.from_documents(docs, embeddings)

    # 儲存 FAISS DB（包含 `.faiss` 和 `.pkl`）
    vector_db.save_local(BaseSettings.vector_index_path)

    print(f"FAISS 向量庫已儲存至: {BaseSettings.vector_index_path}")

# 執行
def run_process():
    # Step 1: 載入 JSON 文件
    data = load_json(BaseSettings.output_file_path)
    
    # Step 2: 建立 FAISS 索引
    save_faiss_index(data)

if __name__ == "__main__":
    run_process()