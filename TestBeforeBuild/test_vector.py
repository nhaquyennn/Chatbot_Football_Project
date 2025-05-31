import os
from dotenv import load_dotenv
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Bước 1: Load API Key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Bước 2: Tạo embedding function giống như khi insert
embedding_function = OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

# Bước 3: Kết nối tới Chroma
chroma_client = PersistentClient(path="../vectorstore/chroma")  # Cập nhật đúng path nhé cưng
collection = chroma_client.get_collection(
    name="sports_data",
    embedding_function=embedding_function
)

# Câu truy vấn test
query = "World Cup là gì?"
print(f"\n📊 Số lượng tài liệu trong collection: {collection.count()}")
print(f"Đang truy vấn với câu: '{query}'")

# Bước 4: Truy vấn
results = collection.query(
    query_texts=[query],
    n_results=3,
    include=["documents", "metadatas", "distances"]  # bỏ "ids" đi nha
)


# Bước 5: In kết quả
print(f"\n🔍 Kết quả truy vấn cho câu: '{query}':")
for i in range(len(results["documents"][0])):
    doc = results["documents"][0][i]
    meta = results["metadatas"][0][i]
    doc_id = results["ids"][0][i]
    distance = results["distances"][0][i]
    
    print(f"\n[{i+1}] 📄 ID: {doc_id}")
    print(f"📁 File gốc: {meta.get('filename', 'unknown')}")
    print(f"📏 Khoảng cách (distance): {distance:.4f}")
    print("📝 Nội dung:")
    print(doc)
