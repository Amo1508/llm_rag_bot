from configs.settings import BaseSettings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from configs.settings import BaseSettings, OllamaSettings
from langchain.prompts import PromptTemplate

# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name=BaseSettings.vector_model)

# 載入 FAISS 向量資料庫（啟用 pickle 反序列化）
vector_db = FAISS.load_local(
    BaseSettings.vector_index_path,
    embedding_model,
    allow_dangerous_deserialization=True
)

# 自定義 Prompt 模板
def get_custom_prompt():
    """
    創建自定義 Prompt 模板，提供更好的指導給 AI
    
    返回:
        PromptTemplate: 自定義 Prompt 模板
    """
    template = """
    你是一個專業的 AI 助手，基於提供的資料庫知識回答問題。

    請遵循以下原則：
    1. 仔細分析所提供的上下文資訊
    2. 提供準確、相關且有幫助的回答
    3. 如果上下文中沒有足夠資訊回答問題，請誠實說明你不知道，不要編造答案
    4. 回答應該簡潔明瞭，但要包含足夠的細節
    5. 只能使用繁體中文回答

    聊天歷史：
    {chat_history}

    上下文資訊：
    {context}

    問題：{question}

    回答：
    """
    return PromptTemplate.from_template(template)

# 建立 RAG Chain
def create_rag_chain():
    """
    建立 RAG Chain

    返回:
        ConversationalRetrievalChain: RAG Chain 物件
    """
    # 使用 LangChain Retriever
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 加入 Chat 記憶
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 建立 RAG Chain
    llm = Ollama(model=OllamaSettings.model_llama_3_1, base_url=OllamaSettings.service_url)
    
    # 使用自定義 Prompt
    custom_prompt = get_custom_prompt()

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    return qa_chain

# 開始聊天
def chat_with_ai(qa_chain, question):
    """
    使用 RAG Chain 進行問答

    參數:
        qa_chain: RAG Chain 物件
        question: 查詢文本

    返回:
        str: AI 回答
    """
    response = qa_chain.invoke({"question": question})
    return response["answer"]

# 執行
def run_process():
    qa_chain = create_rag_chain()
    while True:
        query = input("\n💬 你想問什麼？（輸入 'exit' 結束）：")
        if query.lower() == "exit":
            break
        answer = chat_with_ai(qa_chain, query)
        print("\n🤖 AI 回答：", answer)

if __name__ == "__main__":
    run_process()