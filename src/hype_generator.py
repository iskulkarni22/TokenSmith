"""
hype_generator.py

Hypothetical Prompt Embeddings (HyPE) question generation.
Generates hypothetical student questions for textbook chunks to improve retrieval.
"""

import textwrap
from typing import List
from src.generator import ANSWER_START, ANSWER_END, run_llama_cpp, text_cleaning, get_llama_model


def format_question_generation_prompt(chunk_text: str, num_questions: int = 3) -> str:
    """
    Create prompt for generating hypothetical student questions.
    
    Args:
        chunk_text: The textbook chunk to generate questions for
        num_questions: Number of questions to generate
        
    Returns:
        Formatted prompt string
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful assistant that generates study questions.
        <|im_end|>
        <|im_start|>user
        Read this textbook excerpt and write {num_questions} study questions about it. Each question on a new line.
        
        Textbook excerpt:
        {chunk_text}
        <|im_end|>
        <|im_start|>assistant
        """)
    
    return text_cleaning(prompt)


def generate_questions_for_chunk(
    chunk_text: str,
    model_path: str,
    num_questions: int = 3,
    max_tokens: int = 100,
    temperature: float = 0.5,
    **llm_kwargs
) -> List[str]:
    """
    Generate hypothetical questions for a single chunk.
    
    Args:
        chunk_text: The textbook chunk
        model_path: Path to GGUF model
        num_questions: Number of questions to generate
        max_tokens: Maximum tokens for generation
        temperature: Sampling temperature
    Returns:
        List of generated questions (may be fewer than num_questions if parsing fails)
    """
    # Skip very short chunks
    if len(chunk_text.strip()) < 100:
        return []
    
    try:
        # Generate prompt
        prompt = format_question_generation_prompt(chunk_text, num_questions)
        
        # Generate questions using run_llama_cpp (same as query_enhancement.py)
        response = run_llama_cpp(
            prompt,
            model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            **llm_kwargs
        )
        
        # IMPORTANT: Reset the LLM cache to avoid context bleeding between chunks
        try:
            model = get_llama_model(model_path)
            model.reset()  # Clear KV cache
        except Exception:
            pass  # Ignore if reset fails
        
        # Parse response
        generated_text = response["choices"][0]["text"].strip()
        
        # Print raw output for debugging (first 3 chunks only)
        if not hasattr(generate_questions_for_chunk, '_debug_count'):
            generate_questions_for_chunk._debug_count = 0
        if generate_questions_for_chunk._debug_count < 3:
            # print(f"\n[DEBUG] Raw LLM output:\n{generated_text}\n")
            generate_questions_for_chunk._debug_count += 1
        
        # Return empty if no output
        if not generated_text:
            return []
        
        import re
        
        # Remove common artifacts
        generated_text = generated_text.replace('>>>', '').strip()
        
        # Try multiple splitting strategies
        candidates = []
        
        # Strategy 1: Split by newlines
        newline_split = [l.strip() for l in generated_text.split('\n') if l.strip()]
        candidates.append(newline_split)
        
        # Strategy 2: Split by numbered patterns (1., 2., Q1:, etc.)
        numbered_split = re.split(r'\s*(?:\d+\.\s+|Q\d+:?\s+)', generated_text)
        numbered_split = [q.strip() for q in numbered_split if q.strip()]
        candidates.append(numbered_split)
        
        # Strategy 3: Split by dash patterns (- Question1 - Question2)
        dash_split = re.split(r'\s*-\s+', generated_text)
        dash_split = [q.strip() for q in dash_split if q.strip()]
        candidates.append(dash_split)
        
        # Strategy 4: Split by question marks (for concatenated questions)
        # Look for pattern: "Question1?Question2?" -> split and keep the ?
        qmark_split = []
        if '?' in generated_text:
            parts = generated_text.split('?')
            for part in parts:
                part = part.strip()
                if len(part) > 10:  # Must be substantial
                    qmark_split.append(part + '?')
        candidates.append(qmark_split)
        
        # Use the strategy that gives us closest to num_questions (but at least 1)
        best_split = max(candidates, key=lambda x: len(x) if len(x) <= num_questions else 0)
        if not best_split:
            best_split = max(candidates, key=len)  # Fallback to most splits
        
        questions = []
        for line in best_split:
            # Remove leading numbering, dashes, or periods
            line = re.sub(r'^[\d+.):)\-\.\s]+', '', line).strip()
            
            # Must be substantial (>10 chars)
            if len(line) < 10:
                continue
            
            # Add question mark if missing
            if not line.endswith('?'):
                line = line.rstrip('.!') + '?'
            
            questions.append(line)
        
        return questions[:num_questions]
    
    except Exception as e:
        print(f"Warning: Failed to generate questions for chunk: {e}")
        return []


def generate_questions_batch(
    chunks: List[str],
    model_path: str,
    num_questions: int = 3
) -> List[List[str]]:
    """
    Generate questions for multiple chunks sequentially.
    
    Args:
        chunks: List of textbook chunks
        model_path: Path to GGUF model
        num_questions: Number of questions per chunk
        
    Returns:
        List of question lists (one per chunk)
    """
    all_questions = []
    
    for chunk in chunks:
        questions = generate_questions_for_chunk(chunk, model_path, num_questions)
        all_questions.append(questions)
    
    return all_questions
