from query import answer_question, rerank_documents, FINAL_TOP_K
from langchain_core.documents import Document

def test_query_output_format():
    """Test that the response has the correct structure and required fields."""
    result = answer_question("What is the policy?")
    assert "user_question" in result
    assert "system_answer" in result
    assert "chunks_related" in result
    assert isinstance(result["chunks_related"], list)

def test_response_data_types():
    """Test that all response fields have the correct data types."""
    result = answer_question("How do I reset my password?")
    
    assert isinstance(result["user_question"], str)
    assert isinstance(result["system_answer"], str)
    assert isinstance(result["chunks_related"], list)
    
    # Verify user_question matches input
    assert result["user_question"] == "How do I reset my password?"

def test_chunks_structure():
    """Test that chunks have the correct structure with source and content fields."""
    result = answer_question("What are the vacation policies?")
    
    assert len(result["chunks_related"]) > 0, "Should retrieve at least one chunk"
    assert len(result["chunks_related"]) <= FINAL_TOP_K, f"Should retrieve at most {FINAL_TOP_K} chunks after reranking"
    
    for chunk in result["chunks_related"]:
        assert "source" in chunk, "Each chunk should have a 'source' field"
        assert "content" in chunk, "Each chunk should have a 'content' field"
        assert isinstance(chunk["source"], str), "Source should be a string"
        assert isinstance(chunk["content"], str), "Content should be a string"
        assert len(chunk["source"]) > 0, "Source should not be empty"
        assert len(chunk["content"]) > 0, "Content should not be empty"

def test_system_answer_not_empty():
    """Test that the system generates a non-empty answer."""
    result = answer_question("How is overtime calculated?")
    
    assert len(result["system_answer"]) > 0, "System answer should not be empty"
    assert isinstance(result["system_answer"], str), "System answer should be a string"

def test_different_question_types():
    """Test that the system handles different types of questions."""
    questions = [
        "How do I reset my HR portal password?",
        "Can I work overtime without authorization?",
        "How many vacation days can I have per year?",
        "What is the company policy on remote work?",
    ]
    
    for question in questions:
        result = answer_question(question)
        assert result["user_question"] == question
        assert len(result["system_answer"]) > 0
        assert len(result["chunks_related"]) > 0

def test_chunk_source_paths():
    """Test that chunk sources are valid file paths."""
    result = answer_question("What are the benefits?")
    
    for chunk in result["chunks_related"]:
        source = chunk["source"]
        # Source should be a path (may be relative or absolute)
        assert isinstance(source, str)
        # Should contain 'data' in the path (based on project structure)
        assert "data" in source.lower() or source.endswith(".txt")

def test_response_consistency():
    """Test that multiple calls with the same question return consistent structure."""
    question = "What is the payroll policy?"
    
    result1 = answer_question(question)
    result2 = answer_question(question)
    
    # Both should have the same structure
    assert set(result1.keys()) == set(result2.keys())
    assert result1["user_question"] == result2["user_question"] == question
    # Note: answers may differ due to temperature=1, but structure should be consistent
    assert isinstance(result1["system_answer"], str)
    assert isinstance(result2["system_answer"], str)
    assert len(result1["chunks_related"]) == len(result2["chunks_related"])

def test_long_question():
    """Test that the system handles longer, more complex questions."""
    long_question = "Can you explain in detail how the vacation accrual system works, including monthly accrual rates, maximum caps, and how it interacts with local labor laws?"
    
    result = answer_question(long_question)
    assert result["user_question"] == long_question
    assert len(result["system_answer"]) > 0
    assert len(result["chunks_related"]) > 0

def test_short_question():
    """Test that the system handles very short questions."""
    short_question = "Benefits?"
    
    result = answer_question(short_question)
    assert result["user_question"] == short_question
    assert len(result["system_answer"]) > 0
    assert len(result["chunks_related"]) > 0

def test_question_with_special_characters():
    """Test that the system handles questions with special characters."""
    question = "What's the policy on O.T. (overtime)?"
    
    result = answer_question(question)
    assert result["user_question"] == question
    assert len(result["system_answer"]) > 0

def test_reranking_functionality():
    """Test that reranking function works correctly with sample documents."""
    question = "How do I reset my password?"
    
    # Create sample documents
    doc1 = Document(
        page_content="You can reset your password from the login screen.",
        metadata={"source": "data/faq_document.txt"}
    )
    doc2 = Document(
        page_content="Company policies are available in the employee handbook.",
        metadata={"source": "data/policies.txt"}
    )
    doc3 = Document(
        page_content="Password reset requires email verification.",
        metadata={"source": "data/faq_document.txt"}
    )
    
    documents = [doc1, doc2, doc3]
    
    # Test reranking
    reranked = rerank_documents(question, documents, top_k=2)
    
    # Should return at most top_k documents
    assert len(reranked) <= 2, "Reranking should return at most top_k documents"
    assert len(reranked) > 0, "Reranking should return at least one document"
    
    # All returned documents should be from the input list
    reranked_contents = [doc.page_content for doc in reranked]
    original_contents = [doc.page_content for doc in documents]
    for content in reranked_contents:
        assert content in original_contents, "Reranked documents should be from original set"

def test_reranking_empty_documents():
    """Test that reranking handles empty document list gracefully."""
    question = "Test question"
    empty_docs = []
    
    reranked = rerank_documents(question, empty_docs, top_k=3)
    assert reranked == [], "Reranking empty list should return empty list"

def test_final_chunk_count():
    """Test that final chunk count matches FINAL_TOP_K configuration."""
    result = answer_question("What are the benefits?")
    
    # Should retrieve exactly FINAL_TOP_K chunks (or fewer if not enough available)
    assert len(result["chunks_related"]) <= FINAL_TOP_K, \
        f"Should retrieve at most {FINAL_TOP_K} chunks after reranking"
    assert len(result["chunks_related"]) > 0, "Should retrieve at least one chunk"

def test_reranking_improves_relevance():
    """Test that reranking produces consistent results (structure-wise)."""
    question = "How is overtime calculated?"
    
    result1 = answer_question(question)
    result2 = answer_question(question)
    
    # Both should have the same number of chunks (reranking should be consistent)
    assert len(result1["chunks_related"]) == len(result2["chunks_related"]), \
        "Reranking should produce consistent chunk counts"
    
    # Both should have valid chunks
    assert len(result1["chunks_related"]) > 0
    assert len(result2["chunks_related"]) > 0
    
    # All chunks should have source and content
    for chunk in result1["chunks_related"] + result2["chunks_related"]:
        assert "source" in chunk
        assert "content" in chunk

if __name__ == "__main__":
    # Run all tests
    test_query_output_format()
    print("✓ test_query_output_format passed")
    
    test_response_data_types()
    print("✓ test_response_data_types passed")
    
    test_chunks_structure()
    print("✓ test_chunks_structure passed")
    
    test_system_answer_not_empty()
    print("✓ test_system_answer_not_empty passed")
    
    test_different_question_types()
    print("✓ test_different_question_types passed")
    
    test_chunk_source_paths()
    print("✓ test_chunk_source_paths passed")
    
    test_response_consistency()
    print("✓ test_response_consistency passed")
    
    test_long_question()
    print("✓ test_long_question passed")
    
    test_short_question()
    print("✓ test_short_question passed")
    
    test_question_with_special_characters()
    print("✓ test_question_with_special_characters passed")
    
    test_reranking_functionality()
    print("✓ test_reranking_functionality passed")
    
    test_reranking_empty_documents()
    print("✓ test_reranking_empty_documents passed")
    
    test_final_chunk_count()
    print("✓ test_final_chunk_count passed")
    
    test_reranking_improves_relevance()
    print("✓ test_reranking_improves_relevance passed")
    
    print("\n✅ All tests passed!")
