## README

### Image Analyzer API

This FastAPI application provides an endpoint for analyzing images using a pre-trained DenseNet121 model. The API accepts a base64-encoded image and returns the predicted result along with response details.

### Setup

1. Install the required dependencies:
   ```bash
   pip install fastapi[all] torch torchvision Pillow uvicorn
   ```

2. Run the FastAPI application:
   ```bash
   uvicorn your_script_name:app --reload
   ```
   Replace `your_script_name` with the name of the script containing the FastAPI application.

### API Endpoint

- **Endpoint:** `/image-analyzer`
- **Method:** POST
- **Request Body:**
  - `image_base64`: Base64-encoded string representing the image.

### Example Usage

```python
import requests
import base64

# Load image as base64 string
with open("path/to/your/image.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# API Request
url = "http://127.0.0.1:8000/image-analyzer"
payload = {"image_base64": image_base64}
response = requests.post(url, json=payload)

# Print Response
print(response.json())
```

### Response Format

- Successful Response:
  ```json
  {
    "result": 0,
    "response_code": 200,
    "response_message": "Image analysis successful"
  }
  ```

### Custom Exceptions

The API may raise the following custom exceptions:

- `AuthenticationError`: 401 Unauthorized
- `AuthorizationError`: 403 Forbidden
- `BadRequestError`: 400 Bad Request
- `NotFoundError`: 404 Not Found
- `UnprocessableEntityError`: 422 Unprocessable Entity

### Middleware

#### Logging Middleware

The application includes logging middleware that logs each incoming request and its corresponding response.

### CORS Configuration

CORS (Cross-Origin Resource Sharing) is configured to allow all origins (`*`). Update the `origins` list in the code to restrict access to specific domains.

```python
origins = ["http://your-frontend-domain.com"]
```

### Note

- Make sure to replace `"http://your-frontend-domain.com"` with the actual domain of your frontend application in the CORS configuration.

### Dependencies

- `fastapi`: Web framework for building APIs with Python.
- `torch`, `torchvision`: PyTorch and its vision library for deep learning.
- `Pillow`: Python Imaging Library to handle image processing.
- `uvicorn`: ASGI server for running FastAPI applications.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
