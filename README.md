# Agentic RAG Local

Du an RAG cuc bo (local) cho tai lieu PDF, ket hop:
- Embedding Hugging Face (Sentence-Transformers)
- LLM cuc bo qua Ollama
- Giao dien web don gian de hoi dap

## Yeu cau
- Python 3.10+ (khuyen nghi 3.11+)
- Ollama cai san va dang chay
- (Tuy chon) Hugging Face token de tai model nhanh hon

## Cau hinh nhanh
Tao file `.env` (neu can) voi cac bien thong dung:

```
EMBED_BACKEND=hf
EMBED_MODEL=huyydangg/DEk21_hcmute_embedding
EMBED_DEVICE=cpu
RAG_CHUNK_SIZE=1200
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=4
RAG_MAX_CONTEXT_CHARS=4000
```

## Cai dat thu vien
```
pip install -U sentence-transformers torch transformers huggingface-hub pypdf numpy requests
```

Neu gap loi `torchvision::nms`, chay:
```
set TRANSFORMERS_NO_TORCHVISION=1
```

## Tai model embedding (HF)
```
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('huyydangg/DEk21_hcmute_embedding')"
```

Neu bi canh bao ve token, co the dang nhap:
```
huggingface-cli login
```

## Chuan bi du lieu
Dat PDF vao thu muc `document/`.

Chay ingest (incremental):
```
python ingest.py
```

Neu muon re-index toan bo:
```
python ingest.py --reindex-all
```

## Chay server API
Dam bao Ollama dang chay va co model LLM:
```
ollama serve
ollama pull qwen3
```

Chay server:
```
python server.py
```

## Chay Web UI
Mo them terminal khac:
```
cd web
python -m http.server 5500
```

Mo trinh duyet:
```
http://localhost:5500
```

Neu backend khac host/port:
```
http://localhost:5500/?server=http://127.0.0.1:8000
```

## Hoi dap bang CLI
```
python client.py --query "Cau hoi cua ban"
```

## Thu muc quan trong
- `ingest.py`: doc PDF, chunk, embed, luu index/embeddings
- `rag_index.py`: load index, truy van top-k, tao context
- `server.py`: API + CrewAI, goi RAG va LLM
- `rag_data/`: luu index.json, embeddings.npy, state.json
- `document/`: chua PDF
- `web/`: giao dien chat

## Xu ly loi thuong gap
1) **Ollama loi thieu RAM**
   - Loi: `model requires more system memory ...`
   - Cach xu ly: dung model nho hon (1.5B-3B) hoac dong bot ung dung, giam `workers_per_device` trong `server.py`.

2) **Loi torchvision khi dung SentenceTransformers**
   - Dat bien:
     ```
     set TRANSFORMERS_NO_TORCHVISION=1
     ```
   - Hoac go bo torchvision neu khong can:
     ```
     pip uninstall -y torchvision
     ```

3) **Chua thay index**
   - Chay lai:
     ```
     python ingest.py
     ```

## Ghi chu
- He thong uu tien chinh xac: chunk lon + top_k vua phai.
- Neu doi model/parameters, nen chay `ingest.py --reindex-all`.
