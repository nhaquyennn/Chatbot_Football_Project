from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# query = "Ai là vua phá lưới Champions League?"
query = "Bạn có biết Ronaldo không?"

# Prompt dùng cho RAG
# def build_rag_prompt(context):
#     return [
#         {"role": "system", "content": """
#         Bạn là trợ lý AI chỉ được phép trả lời dựa trên dữ liệu bên dưới.
#         - KHÔNG được dùng kiến thức ngoài.
#         - Nếu dữ liệu không đủ để trả lời, hãy nói: 'Không đủ thông tin trong cơ sở dữ liệu.'
#         """.strip()},
#         {"role": "user", "content": f"Dữ liệu:\n{context.strip()}\n\nTrả lời câu hỏi: {query}"}
#     ]
def build_rag_prompt(context):
    return [
        {"role": "system", "content": """
Bạn là một trợ lý AI được yêu cầu chỉ dựa vào dữ liệu được cung cấp để đưa ra câu trả lời.

- Bạn có thể suy luận nếu dữ liệu đủ để làm vậy, nhưng phải NÊU RÕ ĐÂY LÀ SUY LUẬN.
- Nếu dữ liệu KHÔNG CHỨA thông tin rõ ràng, bạn phải nói rõ rằng: 
"Dữ liệu không đề cập trực tiếp, nhưng dựa trên..." nếu có thể suy luận, hoặc "Không đủ thông tin" nếu không thể.
- TUYỆT ĐỐI KHÔNG được dùng kiến thức nền ngoài dữ liệu được cấp.
""".strip()},
        {"role": "user", "content": f"""
Dữ liệu:

{context.strip()}

Hãy trả lời câu hỏi sau một cách chính xác. Nếu phải suy luận, hãy ghi rõ bạn đang suy luận từ đâu.
Câu hỏi: {query}
""".strip()}
    ]


# Trường hợp 1: Không dùng RAG
def no_rag_answer():
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content.strip()

# Trường hợp 2: Dùng RAG nhưng context sai
def rag_wrong_context():
    context = """
    Lionel Messi đã có 120 bàn tại UEFA Champions League.
    Lewandowski cũng là một trong những người ghi nhiều bàn với 91 bàn.
    """
    messages = build_rag_prompt(context)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content.strip()

# Trường hợp 3: Dùng RAG với context đúng
def rag_correct_context():
    context = """
    Cristiano Ronaldo đã ghi 140 bàn tại UEFA Champions League.
    Anh là cầu thủ ghi nhiều bàn thắng nhất trong lịch sử giải đấu.
    """
    messages = build_rag_prompt(context)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("🔎 Câu hỏi:", query)

    print("\n--- Case 1: ❌ Không dùng RAG (GPT thường)")
    print(no_rag_answer())

    print("\n--- Case 2: ⚠️ Dùng RAG nhưng context SAI")
    print(rag_wrong_context())

    print("\n--- Case 3: ✅ Dùng RAG với context ĐÚNG")
    print(rag_correct_context())
