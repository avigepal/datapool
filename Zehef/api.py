import sys
import asyncio
from fastapi import FastAPI
from lib.cli import parser
from lib.colors import *

app = FastAPI()

@app.get("/check_email")
async def check_email(email: str):
    """API Endpoint to check an email"""
    py_version = sys.version_info
    py_require = (3, 10)

    if py_version < py_require:
        return {"error": f"Zehef doesn't work with Python versions lower than 3.10."}

    result = await parser(email)
    return result  # Returns JSON response

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
