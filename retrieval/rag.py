

# test2
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import os
import json
import traceback

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB
base_path = os.path.dirname(os.path.abspath(__file__))
chroma_path = os.path.abspath(os.path.join(base_path, "../vectorstore/chroma"))
chroma_client = chromadb.PersistentClient(path=chroma_path)
collection = chroma_client.get_collection("sports_data")

# Lưu câu hỏi và câu trả lời vào file JSON
def save_to_json(question, answer, filename="chat_history.json"):
    try:
        # Đọc dữ liệu hiện tại trong file (nếu có)
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        # Thêm câu hỏi và câu trả lời vào danh sách
        data.append({"question": question, "answer": answer})

        # Ghi dữ liệu vào file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Đã lưu câu hỏi và câu trả lời vào {filename}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu vào file: {str(e)}")

def get_embedding(query):
    return openai_client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

def hybrid_search_rerank(query, top_k=5):
    if collection.count() == 0:
        return []
    
    # Hiển thị số lượng documents và collections trong ChromaDB
    num_documents = collection.count()
    print(f"Số lượng documents trong collection: {num_documents}")
    
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        query_texts=[query],
        n_results=top_k
    )
    
    docs = zip(results['ids'][0], results['documents'][0], results['distances'][0])
    
    return sorted(docs, key=lambda x: x[2])

# def generate_answer(query, reranked_docs, history=None):
#     if not reranked_docs:
#         return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

#     context = "\n---\n".join([doc for _, doc, _ in reranked_docs])
    
#     # Hiển thị context (dữ liệu được sử dụng để trả lời câu hỏi)
#     print(f"Context tìm thấy từ dữ liệu:\n{context}")

#     # system_prompt = """
#     # Bạn là một hệ thống trả lời câu hỏi chỉ dựa trên dữ liệu được cung cấp.
#     # - KHÔNG sử dụng kiến thức nền
#     # - Hãy duy trì hội thoại liên tục. Nếu người dùng dùng từ như "he", "his", "that player", "anh ấy", "anh ta", "mùa giải đó", "giải đấu trên", v.v.  hãy cố gắng hiểu theo ngữ cảnh từ câu trước đó.
#     # - Nếu không có dữ liệu, trả lời "Không đủ thông tin trong cơ sở dữ liệu."
#     # """
#     system_prompt = """
#     Bạn là một trợ lý AI trả lời câu hỏi về thể thao cụ thể là bóng đá chỉ dựa trên dữ liệu được người dùng cung cấp.
#     - HHãy cố gắng hiểu theo ngữ cảnh từ câu trước đó.
#     - Hãy duy trì hội thoại liên tục. Nếu người dùng dùng từ như "he", "his", "that player", "anh ấy", "anh ta", "mùa giải đó", "giải đấu trên" "trong trận này", v.v. 
#     - Nếu không có dữ liệu, trả lời "Không đủ thông tin trong cơ sở dữ liệu.
#     - Tuyệt đối không sử dụng kiến thức bên ngoài hoặc tự suy luận."
#     """
#     messages = [{"role": "system", "content": system_prompt.strip()}]
#     if history:
#         messages += history[-10:]

#     user_prompt = f"""
#     Dưới đây là dữ liệu:

#     {context}

#     Trả lời câu hỏi: {query}
#     """
#     messages.append({"role": "user", "content": user_prompt.strip()})

#     response = openai_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         temperature=0.7,
#     )
#     return response.choices[0].message.content.strip()
def generate_answer(query, reranked_docs, history=None):
    if not reranked_docs:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

    context = "\n---\n".join([doc for _, doc, _ in reranked_docs])
    
    print(f"Context tìm thấy từ dữ liệu:\n{context}")

    system_prompt = """
    Bạn là một trợ lý AI trả lời câu hỏi về thể thao (đặc biệt là bóng đá) chỉ dựa trên dữ liệu người dùng cung cấp.

    - Bạn không cần phải đề cập đến thời điểm hiện tại của dữ liệu hoặc tình trạng chưa được cập nhật.
    - Nếu câu trả lời không có đủ thông tin, chỉ cần nói "Không đủ thông tin trong cơ sở dữ liệu."
    - Bạn có thể suy luận dựa trên dữ liệu đã cho, nhưng phải nói rõ bạn đang suy luận từ dữ liệu nào.
    - TUYỆT ĐỐI không sử dụng kiến thức nền hoặc thông tin bên ngoài dữ liệu được cung cấp.
    - Hãy duy trì hội thoại liên tục và hiểu các mối liên hệ trong ngữ cảnh.
    """.strip()

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history[-10:]

    user_prompt = f"""
    Dưới đây là dữ liệu:

    {context}

    Hãy trả lời câu hỏi sau. Nếu bạn cần suy luận, hãy ghi rõ là bạn đang suy luận từ dữ liệu nào.

    Câu hỏi: {query}
    """.strip()

    messages.append({"role": "user", "content": user_prompt})

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def handle_query(query, history=None):
    reranked_docs = hybrid_search_rerank(query)
    return generate_answer(query, reranked_docs, history)

# Chỉ chạy hàm main khi file này được gọi trực tiếp, không khi được import
if __name__ == "__main__":
    try:
        # Test câu hỏi và xử lý
        query = "Cầu thủ nào ghi nhiều bàn thắng nhất ở Champions League?"
        answer = handle_query(query)

        # Lưu vào file JSON và in ra terminal
        save_to_json(query, answer)
        print(f"Câu hỏi: {query}")
        print(f"Câu trả lời: {answer}")
    
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {str(e)}")
        traceback.print_exc()
