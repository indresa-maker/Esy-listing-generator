import os
import json
import base64
import mimetypes
import tempfile
import uuid
import asyncio
import requests
import logging
from typing import List, Optional
from io import StringIO

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import aiofiles
import csv
from tenacity import retry, stop_after_attempt, wait_exponential
import boto3
from botocore.exceptions import ClientError

# -------------------- CONFIG --------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# AWS S3 Configuration
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Constants
TAG_MAX_LENGTH = 20
MAX_IMAGE_SIZE_MB = 5
MAX_CONCURRENT_OPENAI = 5
MAX_LISTINGS_PER_REQUEST = 50
OPENAI_TIMEOUT_SECONDS = 60
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_OPENAI)

# -------------------- Pydantic Models --------------------
class ListingOutput(BaseModel):
    SKU: str
    Title: str
    Description: str
    Tags: List[str]

class ErrorResponse(BaseModel):
    SKU: str
    Title: str
    Description: str
    Tags: List[str]
    error: Optional[str] = None

# -------------------- HELPERS --------------------
def sanitize_prompt_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent prompt injection."""
    if not text:
        return ""
    # Remove any control characters and limit length
    sanitized = "".join(char for char in text if ord(char) >= 32 or char in '\n\t')
    return sanitized[:max_length]

def encode_image_base64(path: str) -> str:
    """Encode image file to base64 data URI."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def download_and_encode_image(url: str) -> Optional[str]:
    """Download image from URL and encode to base64."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        mime = response.headers.get("content-type", "image/jpeg")
        b64 = base64.b64encode(response.content).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        return None

def merge_tags(short_keywords, ai_tags):
    """Merge and deduplicate tags, keeping up to 13."""
    all_tags = list(dict.fromkeys(short_keywords + ai_tags))
    return all_tags[:13]

def build_listing_prompt(
    keywords: List[str],
    optional_keywords: List[str],
    notes: str,
    shop_context: str,
    shop_url: str,
    examples_str: str
) -> str:
    """Build the prompt for OpenAI with sanitized inputs."""
    sanitized_notes = sanitize_prompt_input(notes)
    sanitized_shop_context = sanitize_prompt_input(shop_context)
    sanitized_shop_url = sanitize_prompt_input(shop_url)
    
    return f"""
You are an expert Etsy SEO copywriter.

{f'Here are examples of Etsy listings to follow:{examples_str}' if examples_str else ''}

Now analyze the uploaded image and generate Etsy listing content in JSON format:
{{
  "title": "generated title",
  "description": "generated description",
  "tags": ["tag1","tag2",...,"tag20"]
}}

CRITICAL TITLE RULES:
- Title must be 45-85 characters
- Include 2-3 descriptive modifiers
- Do NOT use these long keywords: {json.dumps(optional_keywords)}

DESCRIPTION RULES:
- At least 375 words
- First 300 characters must include all long keywords: {json.dumps(optional_keywords)}
- Additional notes: {sanitized_notes}
- Shop context: {sanitized_shop_context}
- Shop URL: {sanitized_shop_url}

TAG RULES:
- 2-3 words per tag
- 10-20 characters each
- Deduplicate tags
- Maximum 13 tags total

Respond ONLY with valid JSON. Do not include any markdown formatting or code blocks.
"""

async def upload_csv_to_s3(csv_content: str, csv_id: str) -> str:
    """Upload CSV string to S3 and return the URL."""
    try:
        key = f"csvs/{csv_id}.csv"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=csv_content.encode("utf-8"),
            ContentType="text/csv"
        )
        # Return S3 URL
        return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"
    except ClientError as e:
        logger.error(f"Failed to upload CSV to S3: {e}")
        raise HTTPException(status_code=500, detail="Failed to save CSV")

async def save_csv_async(results: List[dict]) -> str:
    """Convert results to CSV string."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["SKU", "Title", "Description", "Tags"])
    
    for item in results:
        tag_string = ", ".join(item.get("Tags", []))
        writer.writerow([
            item.get("SKU", ""),
            item.get("Title", ""),
            item.get("Description", ""),
            tag_string
        ])
    
    return output.getvalue()

# -------------------- OPENAI CALL --------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_openai(prompt: str, image_b64: Optional[str] = None) -> dict:
    """
    Call OpenAI API with retry logic and timeout.
    Supports optional base64 image.
    """
    async with semaphore:
        try:
            # Build content with image if available
            content = [{"type": "text", "text": prompt}]
            if image_b64:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_b64}
                })
            
            # Run API call with timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": content}],
                    temperature=0.7,
                    max_tokens=1500
                ),
                timeout=OPENAI_TIMEOUT_SECONDS
            )

            raw_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)
            
            # Validate response structure
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")
            if "title" not in parsed or "description" not in parsed or "tags" not in parsed:
                raise ValueError("Missing required fields in response")
            if not isinstance(parsed["tags"], list):
                parsed["tags"] = []
                
            return parsed

        except asyncio.TimeoutError:
            logger.warning(f"OpenAI API call timed out after {OPENAI_TIMEOUT_SECONDS}s")
            raise
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse OpenAI response as JSON: {e}\nRaw: {raw_text[:200]}")
            raise
        except ValueError as e:
            logger.warning(f"Invalid response structure: {e}")
            raise
        except Exception as e:
            logger.warning(f"OpenAI API call failed: {e}")
            raise

# -------------------- FASTAPI SETUP --------------------
app = FastAPI(title="Etsy Listing Generator - Production")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# -------------------- HEALTH CHECK --------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "etsy-listing-generator"}

# -------------------- GENERATE LISTINGS WITH IMAGES --------------------
@app.post("/generate_listings_app")
async def generate_listings_app(
    request: str = Form(...),
    images: List[UploadFile] = File(...)
):
    """Generate listings from uploaded images and form data."""
    try:
        data = json.loads(request)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in request: {str(e)}")

    listings = data.get("listings", [])
    examples = data.get("examples", [])
    shop_context = data.get("shop_context", "")
    shop_url = data.get("shop_url", "")

    # Validate batch size
    if len(listings) > MAX_LISTINGS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_LISTINGS_PER_REQUEST} listings per request"
        )

    if len(listings) != len(images):
        raise HTTPException(
            status_code=400,
            detail="Number of listings must match number of images"
        )

    # Build examples string
    examples_str = "\n".join([
        f'Example: title="{sanitize_prompt_input(ex.get("title",""))}", tags={json.dumps(ex.get("tags",[]))}'
        for ex in examples
    ]) if examples else ""

    results = []

    async def process_listing(i: int, listing: dict, image_file: UploadFile):
        sku = listing.get("sku", f"row_{i}")
        image_b64 = None

        # Validate and process image
        if image_file.content_type not in ("image/jpeg", "image/png", "image/webp"):
            logger.warning(f"Invalid image type for SKU {sku}: {image_file.content_type}")
            return {
                "SKU": sku,
                "Title": "",
                "Description": "",
                "Tags": [],
                "error": "Invalid image type"
            }

        try:
            content = await image_file.read()
            
            if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Image too large for SKU {sku}: {len(content)} bytes")
                return {
                    "SKU": sku,
                    "Title": "",
                    "Description": "",
                    "Tags": [],
                    "error": f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit"
                }

            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = os.path.join(temp_dir, image_file.filename or "image.jpg")
                async with aiofiles.open(image_path, "wb") as f:
                    await f.write(content)
                image_b64 = encode_image_base64(image_path)

        except Exception as e:
            logger.error(f"Error processing image for SKU {sku}: {e}")
            return {
                "SKU": sku,
                "Title": "",
                "Description": "",
                "Tags": [],
                "error": str(e)
            }

        # Parse keywords
        raw_keywords = listing.get("keywords", "")
        all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        optional_keywords = [k for k in all_keywords if len(k) > TAG_MAX_LENGTH]
        short_keywords = [k for k in all_keywords if len(k) <= TAG_MAX_LENGTH]

        # Build and call OpenAI
        prompt = build_listing_prompt(
            keywords=short_keywords,
            optional_keywords=optional_keywords,
            notes=listing.get("notes", ""),
            shop_context=shop_context,
            shop_url=shop_url,
            examples_str=examples_str
        )

        try:
            parsed_output = await call_openai(prompt, image_b64=image_b64)
            
            output = ListingOutput(
                SKU=sku,
                Title=parsed_output.get("title", ""),
                Description=parsed_output.get("description", ""),
                Tags=merge_tags(short_keywords, parsed_output.get("tags", []))
            )
            return output.dict()

        except Exception as e:
            logger.error(f"OpenAI processing failed for SKU {sku}: {e}")
            return {
                "SKU": sku,
                "Title": "",
                "Description": "",
                "Tags": [],
                "error": f"AI processing failed: {str(e)}"
            }

    tasks = [process_listing(i, listing, images[i]) for i, listing in enumerate(listings)]
    results = await asyncio.gather(*tasks)

    # Save CSV to S3
    csv_content = await save_csv_async(results)
    csv_id = str(uuid.uuid4())
    csv_url = await upload_csv_to_s3(csv_content, csv_id)

    return JSONResponse({
        "results": results,
        "csv_url": csv_url,
        "success_count": len([r for r in results if "error" not in r or r["error"] is None])
    })

# -------------------- BULK CSV ENDPOINT --------------------
@app.post("/generate_listings_csv")
async def generate_listings_csv(
    file: UploadFile = File(None),
    csv_file: UploadFile = File(None),
    upload: UploadFile = File(None),
    other: UploadFile = File(None)
):
    """Generate listings from CSV with image URLs or paths."""
    csv_upload = next((f for f in [file, csv_file, upload, other] if f and f.filename), None)
    if not csv_upload:
        raise HTTPException(status_code=400, detail="No CSV file uploaded")

    content = await csv_upload.read()
    text = content.decode("utf-8", errors="ignore")

    # Parse CSV
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    listings = []
    
    for idx, row in enumerate(reader):
        sku = row[0].strip() if len(row) > 0 else f"row_{idx}"
        image_url = row[1].strip() if len(row) > 1 else ""
        keywords = row[2].strip() if len(row) > 2 else ""
        notes = row[3].strip() if len(row) > 3 else ""
        
        listings.append({
            "sku": sku,
            "image_url": image_url,
            "keywords": keywords,
            "notes": notes
        })

    # Validate batch size
    if len(listings) > MAX_LISTINGS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_LISTINGS_PER_REQUEST} listings per request"
        )

    results = []

    async def process_csv_listing(i: int, listing: dict):
        sku = listing.get("sku")
        image_b64 = None

        # Download and encode image from URL
        if listing.get("image_url"):
            image_b64 = download_and_encode_image(listing["image_url"])
            if not image_b64:
                logger.warning(f"Failed to download image for SKU {sku}")
                return {
                    "SKU": sku,
                    "Title": "",
                    "Description": "",
                    "Tags": [],
                    "error": "Failed to download image"
                }

        # Parse keywords
        raw_keywords = listing.get("keywords", "")
        all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        optional_keywords = [k for k in all_keywords if len(k) > TAG_MAX_LENGTH]
        short_keywords = [k for k in all_keywords if len(k) <= TAG_MAX_LENGTH]

        prompt = build_listing_prompt(
            keywords=short_keywords,
            optional_keywords=optional_keywords,
            notes=listing.get("notes", ""),
            shop_context="",
            shop_url="",
            examples_str=""
        )

        try:
            parsed_output = await call_openai(prompt, image_b64=image_b64)
            
            output = ListingOutput(
                SKU=sku,
                Title=parsed_output.get("title", ""),
                Description=parsed_output.get("description", ""),
                Tags=merge_tags(short_keywords, parsed_output.get("tags", []))
            )
            return output.dict()

        except Exception as e:
            logger.error(f"OpenAI processing failed for CSV SKU {sku}: {e}")
            return {
                "SKU": sku,
                "Title": "",
                "Description": "",
                "Tags": [],
                "error": f"AI processing failed: {str(e)}"
            }

    tasks = [process_csv_listing(i, listing) for i, listing in enumerate(listings)]
    results = await asyncio.gather(*tasks)

    # Save CSV to S3
    csv_content = await save_csv_async(results)
    csv_id = str(uuid.uuid4())
    csv_url = await upload_csv_to_s3(csv_content, csv_id)

    return JSONResponse({
        "results": results,
        "csv_url": csv_url,
        "success_count": len([r for r in results if "error" not in r or r["error"] is None])
    })

# -------------------- ROOT --------------------
@app.get("/")
def root():
    return {"message": "Etsy Listing Generator backend is running!", "version": "2.0"}
