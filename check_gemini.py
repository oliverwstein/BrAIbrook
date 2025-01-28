#!/usr/bin/env python3
import os
import sys
import time
import google.generativeai as genai
from google.api_core import exceptions

def check_gemini_health():
    """Check if Gemini API is accessible and functioning."""
    
    # First check if API key is set
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        return False

    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Try to initialize both text and vision models
        print("Checking Gemini API status...")
        
        # Test text model
        print("\nTesting text model (gemini-pro)...")
        text_model = genai.GenerativeModel('gemini-pro')
        text_response = text_model.generate_content("Reply with 'ok' if you receive this.")
        if 'ok' in text_response.text.lower():
            print("✅ Text model is working")
        else:
            print("⚠️ Text model responded but with unexpected content")
        
        print("\nTesting vision models...")
        # Create a simple test image (1x1 pixel)
        from PIL import Image
        test_image = Image.new('RGB', (1, 1), color='red')

        vision_models = [
            'gemini-1.5-pro-exp-0827',
            'gemini-1.5-flash',
        ]

        for model_name in vision_models:
            try:
                print(f"\nTesting {model_name}...")
                vision_model = genai.GenerativeModel(model_name)
                vision_response = vision_model.generate_content([
                    "What color is this 1x1 pixel?",
                    test_image
                ])
                print(f"✅ {model_name} is working")
                break
            except exceptions.InternalServerError:
                print(f"❌ {model_name} is experiencing internal issues (500 error)")
            except Exception as e:
                print(f"❌ {model_name} error: {str(e)}")
        
    except exceptions.PermissionDenied:
        print("❌ API key is invalid or has insufficient permissions")
        return False
    except exceptions.ResourceExhausted:
        print("❌ API quota exceeded or rate limit reached")
        return False
    except exceptions.InternalServerError:
        print("❌ Gemini service is experiencing internal issues (500 error)")
        return False
    except exceptions.ServiceUnavailable:
        print("❌ Gemini service is temporarily unavailable (503 error)")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    success = check_gemini_health()
    sys.exit(0 if success else 1)