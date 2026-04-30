#!/usr/bin/env python3
"""
index_updater.py
Adds new chapters to an existing index.
"""

import os
import pickle
import pathlib
import json
import re
from typing import List, Dict, Optional

import faiss
from rank_bm25 import BM25Okapi
from src.embedder import SentenceTransformer

from src.preprocessing.chunking import DocumentChunker, ChunkConfig
from src.preprocessing.extraction import extract_sections_from_markdown
from src.index_builder import build_index, preprocess_for_bm25

DEFAULT_EXCLUSION_KEYWORDS = ['questions', 'exercises', 'summary', 'references']

def add_to_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    embedding_model_context_window: int,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    chapters_to_add: List[int],
    use_multiprocessing: bool = False,
    use_headings: bool = False,
    use_hype: bool = False,
    hype_config: Optional[Dict] = None,
) -> None:
    """
    Adds new chapters to an existing FAISS and BM25 index.
    If an index does not exist, it creates a new one.
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    faiss_index_path = artifacts_dir / f"{index_prefix}.faiss"
    bm25_index_path = artifacts_dir / f"{index_prefix}_bm25.pkl"
    chunks_path = artifacts_dir / f"{index_prefix}_chunks.pkl"
    sources_path = artifacts_dir / f"{index_prefix}_sources.pkl"
    meta_path = artifacts_dir / f"{index_prefix}_meta.pkl"
    info_path = artifacts_dir / f"{index_prefix}_info.json"
    map_path = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    vector_map_path = artifacts_dir / f"{index_prefix}_vector_to_chunk_map.pkl"

    if not faiss_index_path.exists():
        print("No existing index found. Building a new one...")
        build_index(
            markdown_file=markdown_file,
            chunker=chunker,
            chunk_config=chunk_config,
            embedding_model_path=embedding_model_path,
            embedding_model_context_window=embedding_model_context_window,
            artifacts_dir=artifacts_dir,
            index_prefix=index_prefix,
            use_multiprocessing=use_multiprocessing,
            use_headings=use_headings,
            use_hype=use_hype,
            hype_config=hype_config,
            chapters_to_index=chapters_to_add,
        )
        return

    print(f"Adding chapters {chapters_to_add} to existing index...")

    # Load existing artifacts
    with open(chunks_path, "rb") as f:
        existing_chunks = pickle.load(f)
    with open(sources_path, "rb") as f:
        existing_sources = pickle.load(f)
    with open(meta_path, "rb") as f:
        existing_metadata = pickle.load(f)
    with open(info_path, "r") as f:
        index_info = json.load(f)
    
    # Load existing vector-to-chunk map if it exists (for HyPE)
    if vector_map_path.exists():
        with open(vector_map_path, "rb") as f:
            existing_vector_map = pickle.load(f)
    else:
        existing_vector_map = list(range(len(existing_chunks)))  # Identity mapping

    if map_path.exists():
        with open(map_path, "r") as f:
            page_to_chunk_ids = {int(k): set(v) for k, v in json.load(f).items()}
    else:
        page_to_chunk_ids = {}

    target_textbook = next((t for t in index_info["textbooks"] if t["markdown_file"] == markdown_file), None)

    if target_textbook:
        existing_chapters = target_textbook.get("chapters", [])
    else:
        existing_chapters = []

    if "all" in existing_chapters:
        print("Index contains all chapters. No new chapters to add.")
        return

    chapters_to_index = list(set(chapters_to_add) - set(existing_chapters))

    if not chapters_to_index:
        print("All requested chapters are already in the index.")
        return

    # Extract sections for the new chapters
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS
    )
    new_sections = [s for s in sections if s.get('chapter') in chapters_to_index]

    if not new_sections:
        print("No new sections found for the given chapters.")
        return

    new_chunks: List[str] = []
    new_sources: List[str] = []
    new_metadata: List[Dict] = []

    total_chunks = len(existing_chunks)
    current_page = max([m.get('page_number', 1) for m in existing_metadata]) if existing_metadata else 1
    heading_stack = []

    page_pattern = re.compile(r'--- Page (\d+) ---')

    for i, c in enumerate(new_sections):
        current_level = c.get('level', 1)
        chapter_num = c.get('chapter', 0)

        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()

        if c['heading'] != "Introduction":
            heading_stack.append((current_level, c['heading']))

        path_list = [h[1] for h in heading_stack]
        full_section_path = " ".join(path_list)
        full_section_path = f"Chapter {chapter_num} " + full_section_path

        sub_chunks = chunker.chunk(c['content'])

        for sub_chunk_id, sub_chunk in enumerate(sub_chunks):
            chunk_pages = set()
            fragments = page_pattern.split(sub_chunk)
            current_chunk_global_id = total_chunks + len(new_chunks)

            # If there is content before the first page marker, it belongs to the current_page.
            if fragments[0].strip():
                page_to_chunk_ids.setdefault(current_page, set()).add(current_chunk_global_id)
                chunk_pages.add(current_page)

            # Process new pages found within this sub_chunk.
            for j in range(1, len(fragments), 2):
                try:
                    new_page = int(fragments[j]) + 1
                    if fragments[j + 1].strip():
                        page_to_chunk_ids.setdefault(new_page, set()).add(current_chunk_global_id)
                        chunk_pages.add(new_page)
                    current_page = new_page
                except (IndexError, ValueError):
                    continue

            # Clean sub_chunk by removing page markers
            clean_chunk = re.sub(page_pattern, '', sub_chunk).strip()

            if c["heading"] == "Introduction":
                continue

            meta = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "char_len": len(clean_chunk),
                "word_len": len(clean_chunk.split()),
                "section": c['heading'],
                "section_path": full_section_path,
                "text_preview": clean_chunk[:100],
                "page_numbers": sorted(list(chunk_pages)),
                "chunk_id": current_chunk_global_id
            }

            if use_headings:
                chunk_prefix = f"Description: {full_section_path} Content: "
            else:
                chunk_prefix = ""

            new_chunks.append(chunk_prefix + clean_chunk)
            new_sources.append(markdown_file)
            new_metadata.append(meta)

    # Embed new chunks with optional HyPE support
    print(f"Embedding {len(new_chunks)} new chunks...")
    embedder = SentenceTransformer(embedding_model_path, n_ctx=embedding_model_context_window)
    
    new_vectors = []
    new_vector_map = []
    base_chunk_id = len(existing_chunks)  # Starting chunk ID for new chunks
    
    if use_hype and hype_config:
        print("\n=== HyPE Multi-Vector Indexing Enabled ===")
        print(f"Generating {hype_config.get('num_questions', 3)} questions per chunk...")
        
        from src.hype_generator import generate_questions_for_chunk
        from tqdm import tqdm
        
        # Step 1: Generate questions
        print("Step 1/3: Generating questions for new chunks...")
        all_questions = []
        for chunk_id, chunk_text in tqdm(enumerate(new_chunks), total=len(new_chunks), desc="Generating questions"):
            questions = generate_questions_for_chunk(
                chunk_text,
                hype_config['model_path'],
                hype_config.get('num_questions', 3)
            )
            all_questions.append(questions)
        
        # Step 2: Clear LLM from memory
        print("Step 2/3: Clearing LLM from memory...")
        from src.generator import _LLM_CACHE
        _LLM_CACHE.clear()
        import gc
        gc.collect()
        
        # Step 3: Embed chunks and questions
        print("Step 3/3: Embedding chunks and questions...")
        for local_chunk_id, chunk_text in tqdm(enumerate(new_chunks), total=len(new_chunks), desc="Embedding"):
            global_chunk_id = base_chunk_id + local_chunk_id
            
            # Embed original chunk
            chunk_emb = embedder.encode([chunk_text])[0]
            new_vectors.append(chunk_emb)
            new_vector_map.append(global_chunk_id)
            
            # Embed questions
            questions = all_questions[local_chunk_id]
            if questions:
                question_embs = embedder.encode(questions)
                for q_emb in question_embs:
                    new_vectors.append(q_emb)
                    new_vector_map.append(global_chunk_id)
        
        print(f"Generated {len(new_vectors)} new vectors ({len(new_chunks)} chunks + questions)")
        new_embeddings = np.array(new_vectors, dtype=np.float32)
    else:
        # Standard single-vector per chunk
        new_embeddings = embedder.encode(new_chunks, batch_size=4, show_progress_bar=True)
        new_vector_map = list(range(base_chunk_id, base_chunk_id + len(new_chunks)))

    # Add to FAISS index
    faiss_index = faiss.read_index(str(faiss_index_path))
    faiss_index.add(new_embeddings)
    faiss.write_index(faiss_index, str(faiss_index_path))
    print("Updated FAISS index.")

    # Combine old and new artifacts
    all_chunks = existing_chunks + new_chunks
    all_sources = existing_sources + new_sources
    all_metadata = existing_metadata + new_metadata

    # Re-build BM25 index
    print("Re-building BM25 index...")
    tokenized_chunks = [preprocess_for_bm25(chunk) for chunk in all_chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    with open(bm25_index_path, "wb") as f:
        pickle.dump(bm25_index, f)
    print("Updated BM25 index.")

    # Save updated artifacts
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(sources_path, "wb") as f:
        pickle.dump(all_sources, f)
    with open(meta_path, "wb") as f:
        pickle.dump(all_metadata, f)
    
    # Save updated vector-to-chunk map
    combined_vector_map = existing_vector_map + new_vector_map
    with open(vector_map_path, "wb") as f:
        pickle.dump(combined_vector_map, f)
    
    if use_hype:
        print(f"Updated HyPE multi-vector mapping: {len(combined_vector_map)} vectors → {len(all_chunks)} chunks")

    # Save updated page-to-chunk map
    final_map = {str(page): sorted(list(id_set)) for page, id_set in page_to_chunk_ids.items()}
    with open(map_path, "w") as f:
        json.dump(final_map, f, indent=2)
    print(f"Updated page to chunk ID map: {map_path}")

    # Update index info
    updated_chapters = sorted(list(set(existing_chapters + chapters_to_index)))
    if target_textbook:
        target_textbook["chapters"] = updated_chapters
        target_textbook["status"] = "partial" if "all" not in updated_chapters else "full"
    else:
        index_info["textbooks"].append({
            "markdown_file": markdown_file,
            "chapters": updated_chapters,
            "status": "partial" if "all" not in updated_chapters else "full"
        })

    with open(info_path, "w") as f:
        json.dump(index_info, f, indent=2)

    print(f"Successfully added {len(new_chunks)} new chunks for chapters {chapters_to_index}.")
