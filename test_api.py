from fastapi.testclient import TestClient
from app import app
from PIL import Image
import io
import sys

try:
    client = TestClient(app)
except Exception as e:
    print("TestClient Error:", e)
    sys.exit(1)

def test_classify():
    # Create a dummy black image
    img = Image.new('RGB', (224, 224), color = 'black')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    response = client.post("/classify", files={"file": ("test.png", img_byte_arr, "image/png")})
    print("Status Code:", response.status_code)
    if response.status_code == 200:
        data = response.json()
        print("Category:", data.get('category'))
        print("Base64 Length:", len(data.get('image_base64', '')))
        base64_prefix = data.get('image_base64', '')[:30]
        print("Base64 Prefix:", base64_prefix)
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    test_classify()
