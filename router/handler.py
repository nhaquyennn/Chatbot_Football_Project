# from retrieval.rag import hybrid_search_rerank, generate_answer
# import chromadb

# client = chromadb.PersistentClient(path="../vectorstore/chroma")
# collection = client.get_collection("sports_data")

# def route_message(msg):
#     msg = msg.lower()
#     if any(k in msg for k in ["cầu thủ", "đội bóng", "vị trí"]): return "player"
#     if any(k in msg for k in ["tỉ số", "kết quả", "trận đấu"]): return "score"
#     if any(k in msg for k in ["bàn thắng", "kiến tạo", "thống kê"]): return "stat"
#     if any(k in msg for k in ["dự đoán", "cược", "tối nay"]): return "predict"
#     return "other"

# def handle_query(msg, history):
#     intent = route_message(msg)
#     prompt = {
#         "player": "Bạn là chuyên gia cầu thủ hài hước.",
#         "score": "Bạn là bình luận viên bóng đá vui tính.",
#         "stat": "Bạn là nhà thống kê dí dỏm.",
#         "predict": "Bạn là chuyên gia dự đoán bóng đá vui vẻ.",
#         "other": "Bạn là trợ lý bóng đá thân thiện."
#     }[intent]

#     docs = hybrid_search_rerank(msg, collection)
#     return generate_answer(msg, docs, history, system_prompt=prompt)

#test2
# router/handler.py
from retrieval.rag import hybrid_search_rerank, generate_answer

def handle_query(user_query, chat_history=None):
    """Xử lý câu hỏi và trả về câu trả lời từ hệ thống RAG"""
    docs = hybrid_search_rerank(user_query, top_k=10)
    answer = generate_answer(user_query, docs, history=chat_history)
    return answer
