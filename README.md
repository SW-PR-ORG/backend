# Machine Learning–Based Password Strength Evaluation
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import re
import uvicorn

app = FastAPI()

# Enable CORS ( localhost:3000) 
app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure
class PasswordRequest(BaseModel):
    password: str

def mock_ai_model(password: str):
    
    missing_elements = []
    # Chec for numbers
    if not re.search(r"\d", password):
        missing_elements.append("numbers")
    # Check for special character
    if not re.search(r"[ !@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", password):
        missing_elements.append("special characters")

    if missing_elements:
        recommendation = " and ".join(missing_elements)
        return {
            "is_strong": False,
            "message": f"Weak password. We recommend adding {recommendation} to strengthen it."
        }

    return {
        "is_strong": True,
        "message": "Password is strong!"
    }