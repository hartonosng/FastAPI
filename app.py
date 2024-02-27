from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
import logging
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import densenet121
import asyncio
import uvicorn


app = FastAPI()

# Middleware Logging
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

class LoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response: {response.status_code}")
        return response

app.middleware('http')(LoggingMiddleware())

# CORS
origins = ["*"]  # Update with your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained DenseNet121 model
model = densenet121(pretrained=True)
model.eval()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image)

# Custom Exception Classes
class AuthenticationError(Exception):
    pass

class AuthorizationError(Exception):
    pass

class BadRequestError(Exception):
    pass

class NotFoundError(Exception):
    pass

class UnprocessableEntityError(Exception):
    pass
# Image Analyzer Endpoint
@app.post("/image-analyzer")
async def analyze_image(image_base64: str):
    try:
        # Decode base64 and load image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Preprocess image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = model(preprocessed_image.unsqueeze(0))

        # Extract result and format response
        result = output.argmax().item()
        response_code = 200
        response_message = "Image analysis successful"
        return {
            "result": result,
            "response_code": response_code,
            "response_message": response_message
        }

    except AuthenticationError as auth_error:
        response_code = 401
        response_message = f"Authentication Error: {str(auth_error)}"
        raise HTTPException(status_code=response_code, detail=response_message)

    except AuthorizationError as authz_error:
        response_code = 403
        response_message = f"Forbidden: {str(authz_error)}"
        raise HTTPException(status_code=response_code, detail=response_message)

    except BadRequestError as bad_request_error:
        response_code = 400
        response_message = f"Bad Request: {str(bad_request_error)}"
        raise HTTPException(status_code=response_code, detail=response_message)

    except NotFoundError as not_found_error:
        response_code = 404
        response_message = f"Not Found: {str(not_found_error)}"
        raise HTTPException(status_code=response_code, detail=response_message)

    except UnprocessableEntityError as unprocessable_error:
        response_code = 422
        response_message = f"Unprocessable Entity: {str(unprocessable_error)}"
        raise HTTPException(status_code=response_code, detail=response_message)

    except Exception as e:
        # Handle other errors
        response_code = 500
        response_message = f"Internal Server Error: {str(e)}"
        raise HTTPException(status_code=response_code, detail=response_message)

# # Run the FastAPI application
# if __name__ == "__main__":
    
#     uvicorn.run(app, host="127.0.0.1", port=8000)
