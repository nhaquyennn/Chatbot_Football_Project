import chromadb
import json

# Kết nối ChromaDB
client = chromadb.PersistentClient(path="../vectorstore/chroma")

# Lấy collection
try:
    collection = client.get_collection("sports_data")
except Exception as e:
    print(f"❌ Lỗi khi lấy collection: {e}")
    exit(1)

# Lấy tất cả tài liệu
try:
    results = collection.get(
        include=["documents", "metadatas"]
    )
    documents = results["documents"]
    ids = results["ids"]
    metadatas = results["metadatas"]

    if not documents:
        print("⚠️ Collection rỗng, không có tài liệu nào!")
    else:
        print(f"📊 Số lượng tài liệu trong collection: {len(documents)}")
        players_found = False
        for id_, doc, meta in zip(ids, documents, metadatas):
            if meta.get("filename") == "Players_chunked.json":
                players_found = True
                try:
                    # Thử parse JSON nếu tài liệu là chuỗi JSON
                    parsed_doc = json.loads(doc) if doc.startswith("{") or doc.startswith("[") else doc
                    print(f"\n📜 ID: {id_}")
                    print(f"📝 Nội dung: {parsed_doc}")
                    print(f"🔍 Metadata: {meta}")
                except json.JSONDecodeError:
                    # Nếu không phải JSON, in nguyên văn
                    print(f"\n📜 ID: {id_}")
                    print(f"📝 Nội dung: {doc}")
                    print(f"🔍 Metadata: {meta}")
        
        if not players_found:
            print("⚠️ Không tìm thấy tài liệu nào từ file 'players_chunked.json' trong collection!")
except Exception as e:
    print(f"❌ Lỗi khi lấy tài liệu: {e}")