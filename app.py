from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request,Depends, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
import os,html,logging,sys,io,chardet, re, zipfile, asyncio, multiprocessing, csv
from typing import List,Any,Dict,Optional
from phone_utils import normalize_phone_number, clean_mongo_data
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

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


@limiter.limit("10/minute")
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

        # Columns detection
        header_mapping = {
            "phone": ["phone", "Phone", "PHONE", "Contact", "Contact Number", "contact", "contact number"],
            "email": ["email", "Email", "EMAIL", "Email Address", "email address"],
            "name": ["name", "Name", "NAME", "Full Name", "full name"]
        }

        phone_column = next((col for col in df.columns if col.strip() in header_mapping["phone"]), None)
        email_column = next((col for col in df.columns if col.strip() in header_mapping["email"]), None)
        name_column = next((col for col in df.columns if col.strip() in header_mapping["name"]), None)

        if not any([phone_column, email_column, name_column]):
            logger.exception("No valid header found in file")
            raise HTTPException(status_code=400, detail="No valid header (phone, email, name) found in file.")

        # Extract values
        phone_values = df[phone_column].dropna().astype(str).unique().tolist() if phone_column else []
        email_values = df[email_column].dropna().astype(str).unique().tolist() if email_column else []
        name_values = df[name_column].dropna().astype(str).unique().tolist() if name_column else []

        # Normalize phone numbers
        normalized_phone_values = set()
        for val in phone_values:
            try:
                full_number, without_code = normalize_phone_number(val)
                normalized_phone_values.update({str(full_number), str(without_code)})
            except Exception:
                continue

        # Prepare query
        or_conditions = []
        if normalized_phone_values:
            or_conditions.append({"phone": {"$in": list(normalized_phone_values)}})
            or_conditions.append({"phone": {"$in": phone_values}})
        elif email_values:
            or_conditions.append({"email": {"$in": email_values}})
        elif name_values:
            or_conditions.append({"name": {"$in": name_values}})

        if not or_conditions:
            logger.error("No phone or email or name values to search")
            raise HTTPException(status_code=400, detail="No valid data (phone/email/name) to search.")

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

        # Find unmatched entries
        found_phones = set(matches_df["phone"].astype(str)) if "phone" in matches_df.columns else set()
        found_emails = set(matches_df["email"].astype(str)) if "email" in matches_df.columns else set()
        found_names = set(matches_df["name"].astype(str)) if "name" in matches_df.columns else set()

        unmatched_data = {}
        if phone_values:
            unmatched_data["Phone Not Found"] = [val for val in phone_values if val not in found_phones]
        if email_values:
            unmatched_data["Email Not Found"] = [val for val in email_values if val not in found_emails]
        if name_values:
            unmatched_data["Name Not Found"] = [val for val in name_values if val not in found_names]

        no_matches_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in unmatched_data.items()]))

        # Save processed output
        base_filename = os.path.splitext(file.filename)[0].replace(" ", "_").replace("/", "_")
        output_file_name = f"{base_filename}_processed.csv"
        output_file_path = os.path.join(PROCESSED_DIR, output_file_name)

        try:
            if not matches_df.empty:
                matches_df = matches_df[matches_df["phone"].notna() & (matches_df["phone"] != '')]
                matches_df.to_csv(output_file_path, matches_df)
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

# Directories
OUTPUT_DIR = os.getenv("OUTPUT_DIRECTORY")
ZIP_DIR = os.getenv("ZIP_DIRECTORY")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ZIP_DIR, exist_ok=True)

executor = ProcessPoolExecutor()

def process_file_in_process(filename: str, matches: List[dict]) -> Optional[str]:
    try:
        output_df = pd.DataFrame(matches)
        
        if 'phone' not in output_df.columns:
            return None

        output_df = output_df[['name', 'phone']]
        output_df = output_df[
            output_df['phone'].notnull() &
            output_df['phone'].astype(str).str.strip().ne('') &
            output_df['phone'].astype(str).str.len().between(7, 15)
        ]
        output_df.drop_duplicates(subset='phone', inplace=True)

        if output_df.empty:
            return None

        output_filename = f"{os.path.splitext(filename)[0]}_output.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        output_df.to_csv(output_path, index=False)
    
        return output_path
    except Exception as e:
        return None

def normalize_name(name: str) -> str:
    return re.sub(r'\s+', ' ', name.strip().lower())

async def process_full_file(file: UploadFile,unmatched_names_global: set, all_names_global: set, matched_names_global: set) -> Optional[str]:
    try:
        filename = file.filename
        if not (filename.lower().endswith('.csv') or filename.lower().endswith('.xlsx')):
            logger.warning(f"Invalid file type {filename} - Skipping.")
            return None

        contents = await file.read()

        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents))
            else:
                df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        except Exception as e:
            logger.warning(f"Failed to read file {filename}: {e}")
            return None

        if df.empty:
            logger.warning(f"{filename} is empty. Skipping.")
            return None

        df.columns = [col.lower().strip() for col in df.columns]
        
        if 'name' not in df.columns:
            logger.warning(f"'name' column missing in {filename}")
            return None

        # names = [name.strip().lower() for name in df['name'].dropna()]
        names = [normalize_name(name) for name in df['name'].dropna()]
        unique_names = set(names)
        all_names_global.update(unique_names)
        
        if not unique_names:
            logger.warning(f"No valid names found in {filename}")
            return None

        results_cursor = collection.find(
            {"name": {"$in": list(unique_names)}},
            {"name": 1, "phone": 1, "_id": 0}
        )
        results = await results_cursor.to_list(length=None)
        
        if not results:
            logger.info(f"No matches found in DB for {filename}.")
            unmatched_names_global.update(unique_names)
            return None
        
        matched_names = set(normalize_name(entry['name']) for entry in results if 'name' in entry)
        matched_names_global.update(matched_names)# added
        
        unmatched_names = unique_names - matched_names
        unmatched_names_global.update(unmatched_names) # added
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, process_file_in_process, filename, results)

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
        return None
    
async def limited_gather(tasks, limit):
    semaphore = asyncio.Semaphore(limit)
    async def sem_task(task):
        async with semaphore:
            return await task
    return await asyncio.gather(*(sem_task(t) for t in tasks))


@app.post("/api/upload-files/", dependencies=[Depends(verify_internal_request)])
@limiter.limit("10/minute")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    if not files:
        logger.warning("No files provided!")
        return JSONResponse({"error": "No files provided"}, status_code=400)
    logger.info(f"[INTERNAL] Received {len(files)} files from {request.client.host}")

    try:
        unmatched_names_global = set()
        matched_names_global = set()
        all_names_global = set()
        
        tasks = [process_full_file(file, unmatched_names_global, all_names_global,matched_names_global) for file in files]
        dynamic_limit = min(max(2, multiprocessing.cpu_count() // 2), 10)
        completed_paths = await limited_gather(tasks, limit=dynamic_limit)
        output_files = [path for path in completed_paths if path]
        
        unmatched_csv_path = None
        if unmatched_names_global:
            unmatched_filename = f"unmatched_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            unmatched_csv_path = os.path.join(OUTPUT_DIR, unmatched_filename)

            with open(unmatched_csv_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["unmatched_name"])
                for name in unmatched_names_global:
                    writer.writerow([name])
                writer.writerow([])  # blank line before summary
                writer.writerow(["SUMMARY"])
                writer.writerow(["Total Unique Names", len(all_names_global)])
                writer.writerow(["Matched Names", len(matched_names_global)])
                writer.writerow(["Unmatched Names", len(unmatched_names_global)])

            output_files.append(unmatched_csv_path)

        if not output_files:
            logger.warning("All files skipped or produced empty results.")
            return JSONResponse({"error": "No valid matches found."}, status_code=400)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        zip_filename = f"output_{timestamp}.zip"
        zip_path = os.path.join(ZIP_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            for file_path in output_files:
                zipf.write(file_path, arcname=os.path.basename(file_path))
                os.remove(file_path)

        base_url = str(request.base_url).rstrip("/")
        download_link = f"{base_url}/api/download/{zip_filename}"

        logger.info(f"ZIP created successfully: {zip_filename} with {len(output_files)} output files")
        
        summary = {
            "total_names": len(all_names_global),
            "matched_names": len(matched_names_global),
            "unmatched_count": len(unmatched_names_global)
        }

        return {"download_link": download_link, "summary": summary}

    except Exception as e:
        import traceback
        logger.error(f"Fatal server error: {traceback.format_exc()}")
        return JSONResponse({"error": "Server error", "details": str(e)}, status_code=500)


@app.get("/api/download/{filename}", dependencies=[Depends(verify_internal_request)])
async def download(filename: str):
    file_path = os.path.join(ZIP_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(
        file_path,
        filename=filename,
        media_type="application/zip"
    )

# @app.delete("/api/delete/{filename}", dependencies=[Depends(verify_internal_request)])
# async def delete_file(filename: str):
#     file_path = os.path.join(ZIP_DIR, filename)
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         logger.info(f"Deleted file: {file_path}")
#         return {"success": True}
#     return JSONResponse({"error": "File not found"}, status_code=404)

logger.info("FastAPI application started")
