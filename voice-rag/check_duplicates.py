from src.ingestion.loader import load_records
from src.chunking.fixed import chunk_fixed
from src.chunking.parent_child import chunk_parent_child
from src.chunking.sentence import chunk_sentences

def check():
    print("Loading records...")
    records = load_records(1000)
    print(f"Loaded {len(records)} records")
    all_ids = set()
    duplicates = set()
    
    for i, record in enumerate(records):
        if i % 100 == 0: print(f"Processing {i}...")
        for passage_index, passage in enumerate(record.passages):
            chunks = []
            chunks.extend(chunk_sentences(passage, strategy="sentence", language=record.language, query_id=record.query_id, row_index=record.row_index, passage_index=passage_index))
            chunks.extend(chunk_fixed(passage, strategy="fixed", language=record.language, query_id=record.query_id, row_index=record.row_index, passage_index=passage_index))
            chunks.extend(chunk_parent_child(passage, language=record.language, query_id=record.query_id, row_index=record.row_index, passage_index=passage_index))
            
            for c in chunks:
                if c.chunk_id in all_ids:
                    duplicates.add(c.chunk_id)
                all_ids.add(c.chunk_id)
    print(f"Total chunks: {len(all_ids)}")
    print(f"Duplicates: {len(duplicates)}")
    if duplicates:
        print(f"Sample duplicates: {list(duplicates)[:5]}")

check()
