from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from configs.settings import BaseSettings
import re
import os
import json

# 載入 PDF 文件
def load_pdf(pdf_path):
    """
    參數:
        pdf_path: 檔案路徑
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 清理每個文檔的內容
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents

# 清理文本
def clean_text(text):
    """
    清理文本中的不需要內容
    此為簡易範例，尚須依照個人需求調整
    
    參數:
        text: 文字
    """
    # 移除特定格式的頁碼和頁數標記
    text = re.sub(r'第\s*\d+\s*頁，共\s*\d+\s*頁', '', text)
    
    # 移除特定格式的著作權聲明
    text = re.sub(r'本文件著作權為.*?所有', '', text)
    
    # 移除多餘的空白行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

# 將特定區域標記為不可分割，保留未來處理精準度
def preprocess_text(text):
    """
    將原始文本中的程式碼區塊和表格標記為不可分割
    此為簡易範例，尚須依照個人需求調整

    參數:
        text: 文字
    """
    
    # 尋找 ```開頭和結尾的程式碼區塊
    code_pattern = r'(```[\s\S]*?```)'
    # 尋找表格（簡單的判斷，基於連續的|字符）
    table_pattern = r'(\|[-:\s|]*\|\n(?:\|.*\|\n)+)'
    
    # 將找到的程式碼區塊和表格替換為特殊標記
    processed_text = text
    
    # 找出所有匹配的程式碼區塊和表格
    all_matches = []
    for pattern in [code_pattern, table_pattern]:
        matches = re.finditer(pattern, text)
        for match in matches:
            all_matches.append((match.start(), match.end(), match.group(0)))
    
    # 按照起始位置排序匹配結果
    all_matches.sort()
    
    # 重建文本，將程式碼區塊和表格替換為不可分割的特殊標記
    if all_matches:
        processed_text = ""
        last_end = 0
        for start, end, content in all_matches:
            processed_text += text[last_end:start]
            # 將找到的內容用特殊字符包裹，讓分割器視為整體
            processed_text += f"<SPECIAL_BLOCK>\n{content}\n</SPECIAL_BLOCK>"
            last_end = end
        processed_text += text[last_end:]
        
    return processed_text

# 分割文件成較小的區塊
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    將文件分割成較小的區塊
    
    參數:
        documents: LangChain 文件對象列表
        chunk_size: 每個區塊的最大字符數
        chunk_overlap: 區塊之間的重疊字符數
        
    返回:
        list: 分割後的文件區塊列表
    """
    
    # 不分割的區塊
    separators = ["<SPECIAL_BLOCK>", "</SPECIAL_BLOCK>",
                  "\n\n", "\n",
                  " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # 避免在特殊區塊中間斷開
        separators=separators,
        # 不在 <SPECIAL_BLOCK> 標記內部分割
        keep_separator=False
    )
    
    # 分割文檔
    chunks = text_splitter.split_documents(documents)
    
    # 移除特殊標記
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("<SPECIAL_BLOCK>\n", "").replace("\n</SPECIAL_BLOCK>", "")
    
    return chunks

# 輸出json文件
def output_json(chunks):
    """
    將chunks輸出成Json文件
    若無相關路徑則自動建立

    參數:
        chunks: 區塊列表
    """
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(BaseSettings.output_file_path), exist_ok=True)
    
    # 將 Document 物件轉換為可序列化的字典
    serializable_chunks = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    # 將分割後的區塊寫入 JSON 檔案
    with open(BaseSettings.output_file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)

    print(f"JSON 文件已輸出至 {BaseSettings.output_file_path}")

# 執行
def run_process():
    # Step 1: 載入 PDF 文件
    documents = load_pdf(BaseSettings.need_to_parse_file_path)

    # Step 3: 預處理文本
    processed_documents = []
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
        processed_documents.append(doc)

    # Step 4: 分割文件
    chunks = split_documents(processed_documents)

    # Step 5: 輸出Json文件
    output_json(chunks)

if __name__ == "__main__":
    run_process()