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

def write_matches_to_excel(matches_df, no_matches_df, output_path):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        chunk_size = 1000
        for i in range(0, len(matches_df), chunk_size):
            chunk_df = matches_df.iloc[i:i + chunk_size]
            chunk_df.to_excel(
                writer,
                sheet_name="Match Found",
                index=False,
                startrow=i,
                header=(i == 0)
            )
        if not no_matches_df.empty:
            no_matches_df.to_excel(writer, sheet_name="No Matches Found", index=False)


@limiter.limit("5/minute")
@app.post("/api/process-file/",dependencies=[Depends(verify_internal_request)])
async def process_file(request: Request, file: UploadFile = File(...)):
    try:
        if not file:
            logger.error("No file uploaded")
            raise HTTPException(status_code=400, detail="No file uploaded")

        logger.info(f"File received from {request.client.host}: {file.filename}")

        if not file.filename.lower().endswith((".xlsx", ".csv")):
            raise HTTPException(status_code=415, detail="Invalid file type. Only .xlsx and .csv allowed")

        file_contents = await file.read()
        if len(file_contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
        file_stream = io.BytesIO(file_contents)

        try:
            if file.filename.endswith(".xlsx"):
                df = pd.read_excel(file_stream, engine="openpyxl", dtype=str, keep_default_na=False)
            else:
                detected_encoding = chardet.detect(file_contents)['encoding'] or "ISO-8859-1"
                file_stream.seek(0)
                df = pd.read_csv(
                    file_stream,
                    dtype=str,
                    encoding=detected_encoding,
                    delimiter=None,
                    on_bad_lines="skip",
                    skip_blank_lines=True,
                    quotechar='"',
                    low_memory=False,
                )
        except Exception as e:
            logger.exception(f"File read error: {str(e)}")
            raise HTTPException(status_code=422, detail="Failed to parse uploaded file")

        if df.empty:
            logger.warning("Uploaded file is empty")
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        logger.info(f"Columns in uploaded file: {df.columns.tolist()}")

        phone_columns = ["phone", "Phone", "PHONE", "Contact", "Contact Number", "contact", "contact number"]
        email_columns = ["email", "Email", "EMAIL", "Email Address", "email address"]
        name_columns = ["name", "Name", "NAME", "Full Name", "full name"]

        phone_column, email_column, name_column = None, None, None

        header_row = df.columns.tolist()

        for col in header_row:
            col = col.strip()
            if col in phone_columns and not phone_column:
                phone_column = col
            if col in email_columns and not email_column:
                email_column = col
            if col in name_columns and not name_column:
                name_column = col

        if not phone_column and not email_column and not name_column:
            logger.exception("No valid header found in file")
            raise HTTPException(status_code=400, detail="No column found in file")

        phone_values = [] 
        email_values = []
        name_values = []

        if phone_column and phone_column in df.columns:
            phone_values = list(set(df[phone_column].astype(str)))
        if email_column and email_column in df.columns:
            email_values = list(set(df[email_column].astype(str)))
        if name_column and name_column in df.columns:
            name_values = list(set(df[name_column].astype(str)))

        # Normalize phone numbers
        normalized_phone_values = set()
        for val in phone_values:
            try:
                full_number, without_code = normalize_phone_number(val)
                normalized_phone_values.add(int(full_number))
                normalized_phone_values.add(int(without_code))
            except Exception:
                continue

        # Prepare query
        or_conditions = []
        if normalized_phone_values:
            or_conditions.append({"phone": {"$in": list(normalized_phone_values)}})
            or_conditions.append({"phone": {"$in": phone_values}})
        if email_values:
            or_conditions.append({"email": {"$in": email_values}})
        if name_values:
            or_conditions.append({"name": {"$in": name_values}})

        if not or_conditions:
            logger.error("No phone or email or name values to search")
            raise HTTPException(status_code=400, detail="No valid data to search")

        query = {"$or": or_conditions}

        # logger.info(f"Prepared MongoDB query: {query}")

        try:
            matches = await collection.find(query, {"_id": 0}).to_list(None)
            matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()
        except Exception:
            logger.exception("MongoDB query error")
            raise HTTPException(status_code=500, detail="Database query failed")

        if not matches_df.empty and "source" in matches_df.columns:
            matches_df.drop(columns=["source"], inplace=True)

        # Find unmatched phones and emails
        found_phone = set(matches_df["phone"].astype(str)) if phone_column and not matches_df.empty and "phone" in matches_df.columns else set()
        found_email = set(matches_df["email"].astype(str)) if email_column and not matches_df.empty and "email" in matches_df.columns else set()
        found_name = set(matches_df["name"].astype(str)) if name_column and not matches_df.empty and "name" in matches_df.columns else set()

        unmatched_phone = [val for val in phone_values if val not in found_phone]
        unmatched_email = [val for val in email_values if val not in found_email]
        unmatched_name = [val for val in name_values if val not in found_name]

        # Build no_matches DataFrame
        no_matches_data = {}
        if unmatched_phone:
            no_matches_data["Phone Not Found"] = unmatched_phone
        if unmatched_email:
            no_matches_data["Email Not Found"] = unmatched_email
        if unmatched_name:
            no_matches_data["Name Not Found"] = unmatched_name

        no_matches_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in no_matches_data.items()]))

        # Generate output file name based on uploaded file name
        original_filename = os.path.splitext(file.filename)[0]  # remove .xlsx or .csv extension
        safe_filename = original_filename.replace(" ", "_").replace("/", "_")  # make filename safe
        output_file_name = f"{safe_filename}_processed.xlsx"
        output_file_path = os.path.join(PROCESSED_DIR, output_file_name)

        try:
            write_matches_to_excel(matches_df,no_matches_df,output_file_path)
        except Exception:
            logger.exception("Error writing output file")
            raise HTTPException(status_code=500, detail="Failed to write output file")

        logger.info(f"Processed Excel file saved at {output_file_path}")

        return JSONResponse(status_code=200, content={
            "status": "success",
            "match_count": len(matches_df),
            "no_match_count": len(no_matches_df),
            "download_url": f"/download-file/{output_file_path}"
        })

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return JSONResponse(status_code=500, content={"status": 500, "error": "Internal Server Error"})


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
