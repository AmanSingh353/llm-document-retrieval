# tests/test_system.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retriever import search_query
import time
import json

def test_basic_functionality():
    """Test basic query-response functionality"""
    test_cases = [
        {
            "query": "What is medical coverage?",
            "expected_keywords": ["medical", "coverage", "health"],
            "should_contain_answer": True
        },
        {
            "query": "How to file a claim?",
            "expected_keywords": ["claim", "file", "submit"],
            "should_contain_answer": True
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"Testing: {test_case['query']}")
        
        start_time = time.time()
        try:
            answer, docs = search_query(test_case["query"])
            response_time = time.time() - start_time
            
            # Check if expected keywords are present
            answer_lower = answer.lower()
            keywords_found = sum(1 for keyword in test_case["expected_keywords"] 
                               if keyword.lower() in answer_lower)
            
            test_result = {
                "query": test_case["query"],
                "answer": answer,
                "response_time": response_time,
                "keywords_found": keywords_found,
                "total_keywords": len(test_case["expected_keywords"]),
                "docs_retrieved": len(docs),
                "success": keywords_found > 0 and len(answer) > 50
            }
            
            results.append(test_result)
            print(f"✅ Success: {test_result['success']}")
            print(f"⏱️ Response time: {response_time:.2f}s")
            print(f"📚 Documents found: {len(docs)}")
            print("---")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                "query": test_case["query"],
                "error": str(e),
                "success": False
            })
    
    return results

def test_performance():
    """Test system performance under load"""
    query = "What is the claim process?"
    num_requests = 5
    
    response_times = []
    
    for i in range(num_requests):
        start_time = time.time()
        try:
            answer, docs = search_query(query)
            response_time = time.time() - start_time
            response_times.append(response_time)
            print(f"Request {i+1}: {response_time:.2f}s")
        except Exception as e:
            print(f"Request {i+1} failed: {str(e)}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"Average response time: {avg_time:.2f}s")
        print(f"Max response time: {max_time:.2f}s")
        print(f"Min response time: {min_time:.2f}s")
    
    return response_times

if __name__ == "__main__":
    print("🧪 Running System Tests...")
    print("=" * 50)
    
    # Test basic functionality
    print("1. Testing Basic Functionality")
    basic_results = test_basic_functionality()
    
    print("\n2. Testing Performance")
    perf_results = test_performance()
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump({
            "basic_tests": basic_results,
            "performance_times": perf_results
        }, f, indent=2)
    
    print(f"\n📊 Results saved to test_results.json")
