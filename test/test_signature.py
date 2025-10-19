# test/test_signature.py
import sys
import os
from unittest.mock import patch

# Add the project root to Python path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.report_generator import generate_inception_report, ContentGenerator
import inspect

def test_content_generator_signature():
    print("\n🧪 Testing ContentGenerator constructor signature...")
    
    try:
        # Check if ContentGenerator accepts stream_callback
        sig = inspect.signature(ContentGenerator.__init__)
        params = list(sig.parameters.keys())
        
        if 'stream_callback' in params:
            print("✅ ContentGenerator accepts stream_callback parameter!")
            
            # Mock the environment variable to avoid API key error
            with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
                # Try creating an instance
                generator = ContentGenerator(
                    location="Test Location",
                    stream_callback=lambda x, y: None
                )
                print("✅ Successfully created ContentGenerator with stream_callback!")
            return True
        else:
            print("❌ ContentGenerator does NOT have stream_callback parameter")
            print(f"Available parameters: {params}")
            return False
            
    except Exception as e:
        print(f"Error testing ContentGenerator: {e}")
        return False

def test_generate_inception_report_signature():
    print("🧪 Testing generate_inception_report function signature...")
    
    try:
        # Try calling with the new parameter
        result = generate_inception_report(
            "test_classified.json",
            "test_output.pdf", 
            "test_ocr.txt",
            "test_session",
            {},
            stream_callback=lambda x, y: None
        )
        print("✅ Function accepts stream_callback parameter!")
        return True
        
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            print("❌ Function does NOT accept stream_callback parameter")
            print(f"Error: {e}")
            return False
        else:
            print(f"Other error: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_function_parameters():
    print("\n📋 Checking function parameters...")
    
    # Check generate_inception_report
    sig1 = inspect.signature(generate_inception_report)
    params1 = list(sig1.parameters.keys())
    print(f"generate_inception_report parameters: {params1}")
    
    # Check ContentGenerator.__init__
    sig2 = inspect.signature(ContentGenerator.__init__)
    params2 = list(sig2.parameters.keys())
    print(f"ContentGenerator parameters: {params2}")
    
    return 'stream_callback' in params1 and 'stream_callback' in params2

if __name__ == "__main__":
    print("🔍 Testing function signatures for streaming support...")
    
    test1 = test_function_parameters()
    test2 = test_content_generator_signature() 
    test3 = test_generate_inception_report_signature()
    
    print(f"\n📊 Results:")
    print(f"Parameter check: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"ContentGenerator test: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"generate_inception_report test: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed! Streaming should work.")
    else:
        print("\n⚠️  Some tests failed. Check your function signatures.")