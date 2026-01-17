from numpy.ma import compressed
from fastapi import FastAPI, HTTPException, Query, Request, Depends, Header
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from motor.motor_asyncio import AsyncIOMotorClient
import os, html, logging, sys, re
from typing import List, Any, Dict, Optional, Set, Union
from phone_utils import normalize_phone_number, clean_mongo_data
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
from slowapi.middleware import SlowAPIMiddleware
import asyncio, csv, io, zipfile
from datetime import datetime


load_dotenv()
API_KEY = os.getenv("API_KEY")
RAPIDAPI_PROXY_SECRET = {
    s.strip() for s in os.getenv("RAPIDAPI_PROXY_SECRET", "").split(",") if s.strip()
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler("app.log"),  # Log to file
    ],
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


async def verify_rapidapi_request(
    request: Request, x_rapidapi_proxy_secret: str = Header(None)
):
    if not x_rapidapi_proxy_secret:
        raise HTTPException(status_code=403, detail="Missing RapidAPI proxy secret")

    if x_rapidapi_proxy_secret not in RAPIDAPI_PROXY_SECRET:
        logger.warning("Invalid X-RapidAPI-Proxy-Secret")
        raise HTTPException(status_code=403, detail="Unauthorized RapidAPI request")
    logger.info("[RapidAPI] Request verified via proxy secret")
    return True


async def verify_internal_request(
    request: Request,
    x_api_key: str = Header(None),
):
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


async def verify_request(
    request: Request,
    x_rapidapi_proxy_secret: str = Header(None),
    x_api_key: str = Header(None),
):
    """Verify request using either RapidAPI or Internal authentication"""

    # Try RapidAPI authentication first
    if x_rapidapi_proxy_secret:
        try:
            await verify_rapidapi_request(request, x_rapidapi_proxy_secret)
            return True
        except HTTPException:
            pass  # Try internal auth if RapidAPI fails

    # Try Internal authentication
    if x_api_key:
        try:
            await verify_internal_request(request, x_api_key)
            return True
        except HTTPException:
            pass

    # If both fail, raise error
    logger.warning("Authentication failed - no valid credentials provided")
    raise HTTPException(
        status_code=403,
        detail="Unauthorized: Provide either x-rapidapi-proxy-secret or x-api-key",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Invalid input: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "status": 400,
            "error": "Invalid input. Please try again with valid input.",
        },
    )


# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# MongoDB Async Configuration
client = AsyncIOMotorClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=5000)
db = client[os.getenv("MONGO_DB")]
collection = db[os.getenv("MONGO_COLLECTION")]


@app.get("/api", dependencies=[Depends(verify_request)])
def home():
    logger.info("API is working!")
    return JSONResponse(
        status_code=200,
        content={"message": "API is working!"},
    )


class SearchResponse(BaseModel):
    status: int
    page: int
    per_page: int
    total: int
    data: List[Dict[str, Any]]


@limiter.limit("30/minute")
@app.get(
    "/api/search/",
    response_model=SearchResponse,
    dependencies=[Depends(verify_internal_request)],
)
async def search(
    request: Request,
    phone: str = Query(
        None,
        min_length=7,
        max_length=15,
        description="Search by phone number, e.g. +1234567890",
    ),
    email: str = Query(
        None, regex=r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", description="Search by email"
    ),
    name: str = Query(None, min_length=2, max_length=100, description="Search by name"),
    username: str = Query(
        None, min_length=3, max_length=100, description="Search by username"
    ),
    page: int = Query(1, ge=1, description="Page number (must be 1 or greater)"),
    per_page: int = Query(
        10, ge=1, le=100, description="Number of results per page (1-100)"
    ),
):

    if phone:
        phone = phone.replace(" ", "+").strip()
        if not PHONE_REGEX.fullmatch(phone):
            return JSONResponse(
                {
                    "status": 400,
                    "error": "Invalid phone number format. Please provide a valid phone number",
                }
            )

    query_conditions = {}
    if not phone and not email and not name and not username:
        logger.error(
            "At least one search parameter (phone, email, username or name) is required"
        )
        return JSONResponse(
            {
                "status": 400,
                "error": "At least one search parameter (phone, email, username or name) is required",
            }
        )

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

    if username:
        username = html.escape(username)
        query_conditions["username"] = {"$eq": username}

    logger.info(f"Search query: {query_conditions}")

    try:
        # Pagination calculations
        skip = (page - 1) * per_page
        total_results = await collection.count_documents(
            query_conditions
        )  # Async count

        results = (
            await collection.find(query_conditions, {"_id": 0})
            .skip(skip)
            .limit(per_page)
            .to_list(length=per_page)
        )

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
            content={
                "status": 500,
                "error": "Internal Server Error. Please try again later.",
            },
        )


class RapidAPIResponse(BaseModel):
    status: int
    data: List[Dict[str, Any]]


class SearchBody(BaseModel):
    phone: str | None = None
    email: str | None = None


@limiter.limit("30/minute")
@app.get(
    "/api/rapidapi/search/",
    response_model=RapidAPIResponse,
    dependencies=[Depends(verify_rapidapi_request)],
)
async def rapidapi_search(request: Request, payload: SearchBody):
    phone = payload.phone
    email = payload.email
    if phone:
        phone = phone.replace(" ", "+").strip()
        if not PHONE_REGEX.fullmatch(phone):
            return JSONResponse(
                {
                    "status": 400,
                    "error": "Invalid phone number format. Please provide a valid phone number",
                }
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

    mongo_query = (
        {"$or": query_conditions} if len(query_conditions) > 1 else query_conditions[0]
    )
    try:
        # No pagination — get all results
        projection = {"_id": 0, "name": 1}
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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZEHEF_PATH = os.path.join(BASE_DIR, "Zehef")


class EmailRequest(BaseModel):
    email: EmailStr


@limiter.limit("20/minute")
@app.post("/api/email_lookup", dependencies=[Depends(verify_rapidapi_request)])
async def email_lookup(request: EmailRequest):
    email = request.email.strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    zehef_result, holehe_result = await asyncio.gather(
        z_search(request), h_search(request)
    )

    used = {}

    IGNORED_KEYS = ["data", "email used, [-] email not used, [x] rate limit"]
    # Zehef
    for service, val in zehef_result.items():
        if service in IGNORED_KEYS:
            continue
        if val is True:
            used[service] = True

    # Holehe
    for service, val in holehe_result.items():
        if service in IGNORED_KEYS:
            continue
        if val is True:
            used.setdefault(service, True)
    return {"email": email, "results": used}


async def z_search(request: EmailRequest):
    email = request.email.strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email is required.")

    sys.path.insert(0, ZEHEF_PATH)

    try:
        from lib.cli import parser
    except ImportError as e:
        logger.critical(f"Failed to import Zehef parser: {e}")
        return {"status": "error", "message": "Zehef parser not found."}

    py_version = sys.version_info
    py_require = (3, 10)

    if py_version < py_require:
        return {
            "status": "error",
            "message": f"Zehef doesn't work with Python versions lower than 3.10.",
        }

    result = await parser(email)
    return result


def normalize_service(service: str) -> str:
    service = service.lower()

    replacements = {
        "twitter.com": "twitter",
        "x (twitter)": "twitter",
        "github.com": "github",
        "gravatar.com": "gravatar",
        "instagram.com": "instagram",
        "facebook.com": "facebook",
    }

    for key, value in replacements.items():
        if key in service:
            return value

    return service.replace(".com", "").strip()


async def h_search(request: EmailRequest):
    email = request.email.strip()
    
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    holehe_path = os.path.join(venv_path, ".venv", "bin", "holehe")
    
    if not os.path.exists(holehe_path):
        logger.error("Holehe executable not found.")
        holehe_path = "holehe"  # Assume it's in PATH

    try:
        proc = await asyncio.create_subprocess_exec(
            holehe_path,
            email,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.error(stderr.decode().strip())
            return {}

        output_lines = stdout.decode().strip().split("\n")

        results: dict[str, bool] = {}

        for line in output_lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith("[+]"):
                service = line[3:].strip()
                results[normalize_service(service)] = True

            elif line.startswith(("[-]", "[x]", "[!]")):
                service = line[3:].strip()
                results[normalize_service(service)] = False

        return results

    except Exception as e:
        logger.exception("Holehe failed")
        return {}


@app.get("/api/telegramsearch", dependencies=[Depends(verify_request)])
async def TelegramSearch(
    request: Request,
    email: Optional[EmailStr] = Query(None, description="Search by email"),
    phone: Optional[str] = Query(None, description="Search by phone number"),
    compressed: bool = Query(True, description="Return compressed ZIP file"),
):
    """
    Search MongoDB by email OR phone (only one at a time) and export results to CSV/ZIP.

    Example usage:
    - /api/search_export?email=user@example.com
    - /api/search_export?phone=1234567890
    - /api/search_export?email=user@example.com&compressed=false
    """

    # Validate that exactly one search parameter is provided
    if (email and phone) or (not email and not phone):
        raise HTTPException(
            status_code=400,
            detail="Please provide exactly one search parameter: either 'email' or 'phone'",
        )

    # Build the search query
    query = {}
    search_type = ""
    search_value = ""

    if email:
        email = html.escape(email)
        query = {"email": email}
        search_type = "email"
        search_value = email
    elif phone:
        phone = html.escape(phone)
        search_type = "phone"
        search_value = phone

        full_number, without_code = normalize_phone_number(phone)
        normalized_value = {full_number, without_code, phone}
        
        try:
            normalized_value.add(int(full_number))
            normalized_value.add(int(without_code))
        except ValueError:
            pass
        
        query = {"phone": {"$in": list(normalized_value)}}
        
    # Search MongoDB
    try:
        cursor = collection.find(query)
        results = await cursor.to_list(length=None)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error, Please try again..: {str(e)}"
        )

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No records found for {search_type}: {search_value}",
        )
        
    serialized_results = [clean_mongo_data(doc) for doc in results]
    return JSONResponse(
            content={
                "success": True,
                "search_type": search_type,
                "search_value": search_value,
                "total_records": len(serialized_results),
                "timestamp": datetime.now().isoformat(),
                "data": serialized_results
            },
            headers={
                "X-Record-Count": str(len(results)),
                "X-Format": "json",
            }
        )


logger.info("FastAPI application started")
