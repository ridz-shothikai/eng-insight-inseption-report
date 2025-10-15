import sys
import os
from unittest.mock import patch, MagicMock
import pytest

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ocr import GoogleCloudVisionOCR, process_ocr, process_single_page_worker
from PIL import Image
import io

def test_ocr_initialization():
    """Test OCR class initialization"""
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
        ocr = GoogleCloudVisionOCR(api_key='test-key', max_workers=2, dpi=150)
        
        assert ocr.api_key == 'test-key'
        assert ocr.max_workers == 2
        assert ocr.dpi == 150
        assert ocr.chunk_size == 50

def test_ocr_initialization_missing_api_key():
    """Test OCR initialization without API key"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Google Cloud Vision API key not found"):
            GoogleCloudVisionOCR()

def test_process_single_page_worker():
    """Test single page worker function"""
    # Create a mock image
    mock_image = Image.new('RGB', (100, 100), color='white')
    page_data = (1, mock_image)
    
    with patch('src.ocr.requests.post') as mock_post:
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "responses": [{
                "fullTextAnnotation": {
                    "text": "Test OCR text from page 1"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        page_num, text = process_single_page_worker(page_data, 'test-key')
        
        assert page_num == 1
        assert text == "Test OCR text from page 1"
        mock_post.assert_called_once()

def test_process_single_page_worker_rate_limit():
    """Test worker retry logic on rate limit"""
    mock_image = Image.new('RGB', (100, 100), color='white')
    page_data = (1, mock_image)
    
    with patch('src.ocr.requests.post') as mock_post:
        # First call: rate limit, second call: success
        mock_post.side_effect = [
            MagicMock(status_code=429),
            MagicMock(status_code=200, json=lambda: {
                "responses": [{"fullTextAnnotation": {"text": "Retry success"}}]
            })
        ]
        
        page_num, text = process_single_page_worker(page_data, 'test-key')
        
        assert page_num == 1
        assert text == "Retry success"
        assert mock_post.call_count == 2

def test_process_single_page_worker_failure():
    """Test worker failure handling"""
    mock_image = Image.new('RGB', (100, 100), color='white')
    page_data = (1, mock_image)
    
    with patch('src.ocr.requests.post') as mock_post:
        mock_post.side_effect = Exception("API error")
        
        page_num, text = process_single_page_worker(page_data, 'test-key')
        
        assert page_num == 1
        assert text == ""


def test_process_ocr_function():
    """Test the main process_ocr function"""
    with patch('src.ocr.GoogleCloudVisionOCR') as mock_ocr_class:
        mock_instance = MagicMock()
        mock_instance.process_document.return_value = True
        mock_ocr_class.return_value = mock_instance
        
        success = process_ocr(
            input_path="test.pdf",
            output_path="output.txt",
            session_id="test-session",
            progress_store={"test-session": {"ocr": 0.0}},
            api_key="test-key"
        )
        
        assert success == True
        mock_ocr_class.assert_called_once_with(api_key='test-key')
        mock_instance.process_document.assert_called_once()

def test_process_ocr_function_failure():
    """Test process_ocr function error handling"""
    with patch('src.ocr.GoogleCloudVisionOCR') as mock_ocr_class:
        mock_ocr_class.side_effect = Exception("OCR initialization failed")
        
        success = process_ocr(
            input_path="test.pdf",
            output_path="output.txt",
            session_id="test-session"
        )
        
        assert success == False

def test_ocr_cleanup():
    """Test OCR cleanup method"""
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test-key'}):
        ocr = GoogleCloudVisionOCR()
        ocr.session = MagicMock()
        
        ocr.cleanup()
        ocr.session.close.assert_called_once()

if __name__ == "__main__":
    # Run the tests
    print("🧪 Running OCR tests...")
    
    test_ocr_initialization()
    print("✅ OCR initialization test passed")
    
    test_ocr_initialization_missing_api_key()
    print("✅ Missing API key test passed")
    
    test_process_single_page_worker()
    print("✅ Single page worker test passed")
    
    test_process_single_page_worker_rate_limit()
    print("✅ Rate limit retry test passed")
    
    test_process_single_page_worker_failure()
    print("✅ Worker failure test passed")
    
    
    test_process_ocr_function()
    print("✅ Process OCR function test passed")
    
    test_process_ocr_function_failure()
    print("✅ Process OCR failure test passed")
    
    test_ocr_cleanup()
    print("✅ OCR cleanup test passed")
    
    print("\n🎉 All OCR tests passed!")