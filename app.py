from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
import os,uuid,html,logging,sys,io,traceback
from typing import List
from phone_utils import normalize_phone_number, clean_mongo_data
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from bson import Binary, ObjectId

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler("app.log")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

# Function to verify API Key
def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=403, detail="Unauthorized")

app = FastAPI(dependencies=[Depends(verify_api_key)])

# Allow only your frontend domain (replace with your actual domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.middleware("http")
async def restrict_ip_access(request: Request, call_next):
    # Get client IP
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    logger.info(f"Request from IP: {client_ip}")
    if client_ip not in os.getenv("ALLOWED_IPS"):
        return JSONResponse({"error": "Unauthorized IP"}, status_code=403)
    return await call_next(request)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Invalid input: {exc}")
    return JSONResponse(
        status_code=400,
        content={"status": 400, "error": "Invalid input. Please try again with valid input."},
    )

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# MongoDB Async Configuration
client = AsyncIOMotorClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
db = client[os.getenv("MONGO_DB")]
collection = db[os.getenv("MONGO_COLLECTION")]

# Ensure processed directory exists
PROCESSED_DIR = os.getenv("PROCESSED_DIR")
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.get("/api/")
def home():
    logger.info("API is working!")
    return JSONResponse(
        status_code=200, content={"message": "API is working!"},
    )

@app.post("/api/process-file/")
@limiter.limit("5/minute")
async def process_file(request: Request,file: UploadFile = File(...)):
    try:
        if not file:
            return JSONResponse({"status": 400, "error": "No file uploaded"})

        logger.info(f"File received: {file.filename}")

        # Ensure file is an Excel file
        if not (file.filename.endswith(".xlsx") or file.filename.endswith(".csv")):
            logger.error("Invalid file type")
            return JSONResponse({"status": 400, "error": "Invalid file type. Only .xlsx and .csv allowed"})

        try:
            # Read file into memory using BytesIO (fix for cPanel issue)
            file_contents = await file.read()
            file_stream = io.BytesIO(file_contents)

            if file.filename.endswith(".xlsx"):
                # df = pd.read_excel(file_stream, engine="openpyxl", dtype=str)
                df = pd.read_csv(
                        file_stream,
                        dtype=str,
                        encoding="utf-8",
                        delimiter=None,
                        on_bad_lines="skip",
                        skip_blank_lines=True,
                        quotechar='"',
                        low_memory=False
                    )
            else:
                df = pd.read_csv(file_stream, dtype=str)
        except Exception as e:
            e_traceback = traceback.format_exc()
            logger.error(f"File read error: {e_traceback}")
            logger.error(f"File read error: {str(e)}")
            return JSONResponse({"status": 500, "error": f"File read error: {str(e)}"})

        logger.info(f"Columns in uploaded file: {df.columns.tolist()}")
        if df.empty:
            logger.error("Uploaded file is empty")
            return JSONResponse({"status": 400, "error": "Uploaded file is empty"})

        phone_columns = ["phone", "PhoneNumber", "Phone", "Phone Number", "Contact", "Mobile"]
        email_columns = ["email", "Email", "Email Address"]
        match_column = None
        search_field = None

        for col in df.columns:
            if col.strip() in phone_columns:
                match_column = col.strip()
                search_field = "phone"
                break

        if not match_column:
            for col in df.columns:
                if col.strip() in email_columns:
                    match_column = col.strip()
                    search_field = "email"
                    break

        if not match_column:
            logger.error("No valid phone or email column found in file")
            return JSONResponse({"status": 400, "error": "No valid phone or email column found in file"})

        # Convert values to string
        values_to_search = df[match_column].astype(str).tolist()
        normalized_values = set()  # Store normalized values

        if search_field == "phone":
            for val in values_to_search:
                full_number, without_code = normalize_phone_number(val)
                normalized_values.add(int(full_number))
                normalized_values.add(int(without_code)) # Ignore if conversion fails
        else:
            normalized_values = set(values_to_search)  # For emails, keep as strings
            values_as_int = set()

        # MongoDB query to match both string & integer formats
        query = {
            "$or": [
                {search_field: {"$in": list(normalized_values)}},
                {search_field: {"$in": list(values_to_search)}},  
            ]
        }

        logger.info(f"Extracted {len(values_to_search)} values from column: {match_column}")

        try:
                # Fetch matching records from MongoDB
                matches = await collection.find(query, {"_id": 0}).to_list(None)
                matches_df = pd.DataFrame(matches)
        except Exception as e:
            error_message = traceback.format_exc()
            logger.error(f"MongoDB query error: {error_message}")
            logger.error(f"MongoDB query error: {str(e)}")
            return JSONResponse({"status": 500, "error": "MongoDB query failed"})
        
        if "source" in matches_df.columns:
            matches_df = matches_df.drop(columns=["source"])
        found_values = set(matches_df[search_field].astype(str).tolist()) if not matches_df.empty else set()

        # Identify non-matching values
        not_found_values = [val for val in values_to_search if val not in found_values]
        no_matches_df = pd.DataFrame({match_column: not_found_values})

        # Generate unique Excel filename
        unique_filename = f"processed_{uuid.uuid4().hex}.xlsx"
        output_file_path = os.path.join(PROCESSED_DIR, unique_filename)

        try:
                with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
                    if not matches_df.empty:
                        matches_df.to_excel(writer, sheet_name="Match Found", index=False)
                    if not no_matches_df.empty:
                        no_matches_df.to_excel(writer, sheet_name="No Matches Found", index=False)
        except Exception as e:
            error_message = traceback.format_exc()
            logger.error(f"File writing error: {error_message}")
            logger.error(f"File writing error: {str(e)}")
            return JSONResponse({"status": 500, "error": "Error writing output file"})

        logger.info(f"Processed Excel file saved at {output_file_path}")
        return JSONResponse(content={"status": "success", "download_url": f"/download-file/{os.path.basename(output_file_path)}"})

    except Exception as e:
        error_message = traceback.format_exc()  # Get full error traceback
        logger.error(f"Unexpected error: {error_message}")
        logger.error(f"Unexpected error: {str(e)}")
        return JSONResponse({"status": 500, "error": "Internal Server Error"})

@app.get("/api/download-file/{filename}")
@limiter.limit("10/minute")
async def download_file(request: Request, filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return JSONResponse({"status": 404, "error": "File not found"})
    logger.info(f"File downloaded: {file_path}")
    return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=filename)


@app.get("/api/search/")
@limiter.limit("30/minute")
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
        
        # Convert non-serializable MongoDB fields to JSON-compatible format
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

logger.info("FastAPI application started")
