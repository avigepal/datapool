from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request,Depends, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
import os,uuid,html,logging,sys,io
from typing import List,Any,Dict
from phone_utils import normalize_phone_number, clean_mongo_data
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import chardet,re
from pydantic import BaseModel

load_dotenv()
API_KEY = os.getenv("API_KEY")
RAPIDAPI_PROXY_SECRET = os.getenv("X_RAPID_KEY") 

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

app = FastAPI()

# Allow only your frontend domain (replace with your actual domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Convert ALLOWED_IPS from .env to a proper list (removes empty values)
allowed_ips = os.getenv("ALLOWED_IPS", "")
allowed_ips = [ip.strip() for ip in allowed_ips.split(",") if ip.strip()]
PHONE_REGEX = re.compile(r"^\+?\d{7,15}$")

async def verify_rapidapi_request(request: Request, x_rapidapi_proxy_secret: str = Header(None)):
    if not x_rapidapi_proxy_secret or x_rapidapi_proxy_secret != RAPIDAPI_PROXY_SECRET:
        logger.warning("Invalid or missing X-RapidAPI-Proxy-Secret header")
        raise HTTPException(status_code=403, detail="Unauthorized RapidAPI request") 
    logger.info(f"[RapidAPI] Request verified via proxy secret")
    return True

async def verify_internal_request(request: Request,x_api_key: str = Header(None),):
    client_ip = request.headers.get("X-Forwarded-For", request.client.host)
    if client_ip:
        client_ip = client_ip.split(",")[0].strip()

    if not x_api_key or x_api_key != API_KEY:
        logger.warning("Invalid or missing API key.")
        raise HTTPException(status_code=403, detail="Invalid API key")

    if client_ip not in allowed_ips:
        logger.warning(f"Unauthorized IP: {client_ip}")
        raise HTTPException(status_code=403, detail="Unauthorized IP")

    logger.info(f"[INTERNAL] Request from allowed IP: {client_ip}")
    return True

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

@app.get("/api/",dependencies=[Depends(verify_internal_request)])
def home():
    logger.info("API is working!")
    return JSONResponse(
        status_code=200, content={"message": "API is working!"},
    )

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE = 1024 * 1024 * MAX_FILE_SIZE_MB

@limiter.limit("5/minute")
@app.post("/api/process-file/",dependencies=[Depends(verify_internal_request)])
async def process_file(request: Request,file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        logger.info(f"File received from {request.client.host}: {file.filename}")

        # Ensure file is an Excel file
        if not file.filename.lower().endswith((".xlsx", ".csv")):
            raise HTTPException(status_code=415, detail="Invalid file type. Only .xlsx and .csv allowed")

        file_contents = await file.read()
        if len(file_contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
        file_stream = io.BytesIO(file_contents)

        try:
            if file.filename.endswith(".xlsx"):    
                # **Use Pandas but with robust error handling**
                df = pd.read_excel(file_stream,engine="openpyxl",dtype=str,keep_default_na=False)
            else:
                detected_encoding = chardet.detect(file_contents)['encoding'] or "ISO-8859-1"
                # ✅ Use detected encoding or fallback to ISO-8859-1
                file_stream.seek(0)
                df = pd.read_csv(
                    file_stream,
                    dtype=str,
                    encoding=detected_encoding,
                    delimiter=None,  # Auto-detect delimiter
                    on_bad_lines="skip",  # Skip corrupt rows
                    skip_blank_lines=True,
                    quotechar='"',
                    low_memory=False,
                )
        except Exception as e:
            logger.exception(f"File read error: {str(e)}")
            raise HTTPException(status_code=422,detail="Failed to parse uploaded file")
                # Ensure DataFrame is not empty
        if df.empty:
            logger.warning("Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        logger.info(f"Columns in uploaded file: {df.columns.tolist()}")

        phone_columns = ["phone", "Phone", "PHONE"]
        email_columns = ["email", "Email", "EMAIL"]
        match_column, search_field = None, None

        for col in df.columns:
            if col.strip() in phone_columns:
                match_column, search_field = col.strip(), "phone"
                break

        if not match_column:
            for col in df.columns:
                if col.strip() in email_columns:
                    match_column, search_field = col.strip(), "email"
                    break

        if not match_column:
            logger.exception("No valid phone or email column found in file")
            raise HTTPException(status_code=400, detail="No valid phone or email column found in file")

        # Convert values to string
        values_to_search = list(set(df[match_column].astype(str)))
        normalized_values = set()

        if search_field == "phone":
            for val in values_to_search:
                try:
                    full_number, without_code = normalize_phone_number(val)
                    normalized_values.add(int(full_number))
                    normalized_values.add(int(without_code))
                except Exception:
                    continue
        else:
            normalized_values = set(values_to_search)

        # MongoDB query to match both string & integer formats
        query = {
            "$or": [
                {search_field: {"$in": list(normalized_values)}},
                {search_field: {"$in": values_to_search}},  
            ]
        }

        logger.info(f"Extracted {len(values_to_search)} values from column: {match_column}")

        try:
            # Fetch matching records from MongoDB
            matches = await collection.find(query, {"_id": 0}).to_list(None)
            matches_df = pd.DataFrame(matches) if matches is not None else pd.DataFrame()
        except Exception:
            logger.exception(f"MongoDB query error")
            raise HTTPException(status_code=500, detail="Database query failed")
        
        if not matches_df.empty and "source" in matches_df.columns:
            matches_df.drop(columns=["source"], inplace=True)

        found_values = set(matches_df[search_field].astype(str)) if not matches_df.empty else set()

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
        except Exception:
            logger.exception("Error writing output file")
            raise HTTPException(status_code=500, detail="Failed to write output file")

        logger.info(f"Processed Excel file saved at {output_file_path}")

        return JSONResponse(status_code=200, content={
            "status": "success",
            "match_count": len(matches_df),
            "no_match_count": len(no_matches_df),
            "download_url": f"/download-file/{os.path.basename(output_file_path)}"
        })

    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500,content={"status": 500, "error": "Internal Server Error"})

@limiter.limit("10/minute")
@app.get("/api/download-file/{filename}",dependencies=[Depends(verify_internal_request)])
async def download_file(request: Request, filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return JSONResponse({"status": 404, "error": "File not found"})
    logger.info(f"File downloaded: {file_path}")
    return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=filename)


class SearchResponse(BaseModel):
    status : int
    page : int
    per_page : int
    total : int
    data : List[Dict[str, Any]]

@limiter.limit("30/minute")
@app.get("/api/search/",response_model=SearchResponse,dependencies=[Depends(verify_internal_request)])
async def search(
    request: Request,
    phone: str = Query(None, min_length=7, max_length=15, description="Search by phone number, e.g. +1234567890"),
    email: str = Query(None, regex=r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", description="Search by email"),
    name: str = Query(None, min_length=2, max_length=100, description="Search by name"),
    page: int = Query(1, ge=1, description="Page number (must be 1 or greater)"),
    per_page: int = Query(10, ge=1, le=100, description="Number of results per page (1-100)"),
):

    if phone:
        phone = phone.replace(" ","+").strip()
        if not PHONE_REGEX.fullmatch(phone):
            return JSONResponse(
                {"status": 400, "error": "Invalid phone number format. Please provide a valid phone number"}
            )

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
       
        results = await collection.find(query_conditions, {"_id":0}).skip(skip).limit(per_page).to_list(length=per_page)
        
        # Sanitize and clean MongoDB data before returning
        results = [clean_mongo_data(doc) for doc in results]

        return {
            "status": 200,
            "page": page,
            "per_page": per_page,
            "total": total_results,
            "data": results,
        }
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "error": "Internal Server Error. Please try again later."}
        )

class RapidAPIResponse(BaseModel):
    status: int
    data: List[Dict[str, Any]]

@limiter.limit("30/minute")
@app.get("/api/rapidapi/search/", response_model=RapidAPIResponse,dependencies=[Depends(verify_rapidapi_request)])
async def rapidapi_search(
    request: Request,
    phone: str = Query(None, min_length=7, max_length=15, description="Search by phone number, e.g. +1234567890"),
    email: str = Query(None, regex=r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", description="Search by email"),
):
    if phone:
        phone = phone.replace(" ","+").strip()
        if not PHONE_REGEX.fullmatch(phone):
            return JSONResponse(
                {"status": 400, "error": "Invalid phone number format. Please provide a valid phone number"}
            )
    if not phone and not email:
        logger.error("[RapidAPI] Either phone or email must be provided")
        return JSONResponse(
            {"status": 400, "error": "Either phone or email is required for search"}
        )

    query_conditions = []
    if phone:
        phone = html.escape(phone)
        full_number, without_code = normalize_phone_number(phone)
        normalized_values = {full_number, without_code}
        try:
            normalized_values.add(int(full_number))
            normalized_values.add(int(without_code))
        except ValueError:
            pass
        query_conditions.append({"phone": {"$in": list(normalized_values)}})

    if email:
        email = html.escape(email)
        query_conditions.append({"email": {"$eq": email}})

    mongo_query = {"$or":query_conditions} if len(query_conditions) > 1 else query_conditions[0]
    logger.info(f"[RapidAPI] Search query: {mongo_query}")

    try:
        # No pagination — get all results
        projection = {"_id": 0, "source": 0}  # Hide source
        results = await collection.find(mongo_query, projection).to_list(length=None)

        cleaned_results = [clean_mongo_data(doc) for doc in results]

        return {
            "status": 200,
            "data": cleaned_results,
        }
    except Exception as e:
        logger.error(f"[RapidAPI] DB Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": 500, "error": "Internal Server Error"},
        )

logger.info("FastAPI application started")
