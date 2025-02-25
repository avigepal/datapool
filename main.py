from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from pymongo import MongoClient
import html
import logging
from bson import Binary, ObjectId

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to MongoDB
MONGO_URI = "your_mongodb_connection_string"
client = MongoClient(MONGO_URI)
db = client["your_database"]
collection = db["your_collection"]

def normalize_phone_number(phone):
    """Normalize phone number by stripping non-numeric characters."""
    digits = "".join(filter(str.isdigit, phone))
    if len(digits) >= 10:
        return digits, digits[-10:]  # Full number and last 10 digits
    return digits, digits

def clean_mongo_data(doc):
    """Convert non-serializable MongoDB fields to JSON-compatible format."""
    for key, value in doc.items():
        if isinstance(value, Binary):
            doc[key] = value.decode("utf-8", errors="ignore")  # Convert Binary to string
        elif isinstance(value, ObjectId):
            doc[key] = str(value)  # Convert ObjectId to string
    return doc

@app.get("/api/search")
async def search(
    request: Request,
    phone: str = Query(None, min_length=7, max_length=15, regex=r"^\+?\d{7,15}$", description="Search by phone number"),
    email: str = Query(None, regex=r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", description="Search by email"),
    name: str = Query(None, min_length=2, max_length=100, description="Search by name"),
    page: int = Query(1, ge=1, description="Page number (must be 1 or greater)"),
    per_page: int = Query(10, ge=1, le=100, description="Number of results per page (1-100)"),
):
    query_conditions = {}

    if not phone and not email and not name:
        logger.error("At least one search parameter (phone, email, or name) is required")
        return JSONResponse({"status": 400, "error": "At least one search parameter (phone, email, or name) is required"})

    if phone:
        phone = html.escape(phone)  # Sanitize input
        full_number, without_code = normalize_phone_number(phone)
        normalized_values = {full_number, without_code}
        try:
            normalized_values.add(int(full_number))
            normalized_values.add(int(without_code))
        except ValueError:
            pass
        query_conditions["phone"] = {"$in": list(normalized_values)}

    if email:
        email = html.escape(email)  # Sanitize input
        query_conditions["email"] = {"$eq": email}  # Prevent regex-based injection

    if name:
        name = html.escape(name)  # Sanitize input
        query_conditions["name"] = {"$eq": name}  # Exact match for name

    logger.info(f"Search query: {query_conditions}")

    try:
        # Pagination calculations
        skip = (page - 1) * per_page
        total_results = await collection.count_documents(query_conditions)  # Async count
        results = await collection.find(query_conditions, {"_id": 0}).skip(skip).limit(per_page).to_list(length=per_page)

        # Convert MongoDB Binary and ObjectId fields to serializable formats
        results = [clean_mongo_data(doc) for doc in results]

        return JSONResponse(content={
            "status": 200,
            "page": page,
            "per_page": per_page,
            "total": total_results,
            "data": results
        })
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "error": "Internal Server Error. Please try again later."}
        )
