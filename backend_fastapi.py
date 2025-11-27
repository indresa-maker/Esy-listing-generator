# backend_fastapi_scaled_full.py
import os
import json
import base64
import mimetypes
import tempfile
import uuid
import asyncio
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import aiofiles
import csv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# -------------------- CONFIG --------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
TAG_MAX_LENGTH = 20
MAX_IMAGE_SIZE_MB = 5
MAX_CONCURRENT_OPENAI = 5  # semaphore limit

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# CSV storage (consider S3 for multi-instance)
csv_store = {}
semaphore = asyncio.Semaphore(MAX_CONCURRENT_OPENAI)

# -------------------- Pydantic Models --------------------
class ListingOutput(BaseModel):
    SKU: str
    Title: str
    Description: str
    Tags: List[str]

# -------------------- HELPERS --------------------
def encode_image_base64(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def merge_tags(short_keywords, ai_tags):
    all_tags = list(dict.fromkeys(short_keywords + ai_tags))  # deduplicate
    return all_tags[:13]
async def save_csv_async(results: List[dict]) -> str:
    filename = f"/tmp/{uuid.uuid4()}.csv"
    async with aiofiles.open(filename, mode="w", encoding="utf-8") as f:
        # Write header
         await f.write("SKU,Title,Description,Tags\n")
    
         # Write each row
        for item in results:
             tag_string = ", ".join(item.get("Tags", []))  # same format you used
             line = (
                f"{item.get('SKU','')},"
                 f"{item.get('Title','')},"
                 f"{item.get('Description','')},"
                f"{tag_string}\n"
            )
            await f.write(line)
    
    return filename

# -------------------- OPENAI CALL --------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def call_openai(prompt: str, image_b64: str = None) -> dict:
    """Call OpenAI API with retry logic"""
    async with semaphore:
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_b64}
            })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}]
        )
        raw_text = response.choices[0].message.content
        try:
            return json.loads(raw_text)
        except Exception as e:
            logger.warning(f"Failed to parse OpenAI response: {e}")
            return {}

# -------------------- FASTAPI SETUP --------------------
app = FastAPI(title="Etsy Listing Generator Scalable")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- APP FORM + IMAGES --------------------
@app.post("/generate_listings_app")
async def generate_listings_app(
    request: str = Form(...),
    images: List[UploadFile] = File(...)
):
    try:
        data = json.loads(request)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    listings = data.get("listings", [])
    examples = data.get("examples", [])
    shop_context = data.get("shop_context", "")
    shop_url = data.get("shop_url", "")

    examples_str = "\n".join([
        f'Example:\n{{"title": "{ex.get("title","")}", "description": "{ex.get("description","")}", "tags": {json.dumps(ex.get("tags",[]))}}}'
        for ex in examples
    ])

    results = []

    async def process_listing(i, listing, image_file):
        sku = listing.get("sku", f"row_{i}")

        if image_file.content_type not in ("image/jpeg", "image/png"):
            logger.warning(f"Invalid image type for SKU {sku}")
            return {"SKU": sku, "Title": "", "Description": "", "Tags": []}
        try:
            # Read the file content
            content = await image_file.read()
    
            # Max file size check
            if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Image too large for SKU {sku}")
                return {"SKU": sku, "Title": "", "Description": "", "Tags": []}
    
            # Save to temporary file
            with tempfile.TemporaryDirectory() as temp_dir:
                image_path = f"{temp_dir}/{image_file.filename}"
                async with aiofiles.open(image_path, "wb") as f:
                    await f.write(content)
                image_b64 = encode_image_base64(image_path)
        
        except Exception as e:
            logger.error(f"Error processing image for SKU {sku}: {e}")
            image_b64 = None

        raw_keywords = listing.get("keywords","")
        all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        optional_keywords_for_ai = [k for k in all_keywords if len(k)>TAG_MAX_LENGTH]
        short_keywords_for_tags = [k for k in all_keywords if len(k)<=TAG_MAX_LENGTH]

        prompt = f"""
You are an expert Etsy SEO copywriter.

Here are examples of Etsy listings to follow:
{examples_str}

Now analyze the uploaded image and generate Etsy listing content in JSON format:
{{
  "title": "generated title",
  "description": "generated description",
  "tags": ["tag1","tag2",...,"tag20"]
}}

CRITICAL TITLE RULES:
- Title 45-85 chars
- Include 2-3 descriptive modifiers
- Exclude long keywords: {json.dumps(optional_keywords_for_ai)}

DESCRIPTION RULES:
- At least 375 words
- First 300 chars must include all long keywords
- Notes: {listing.get('notes','')}
- Shop context: {shop_context}
- Shop URL: {shop_url}

TAG RULES:
- 2-3 words per tag
- 10-20 chars
- Deduplicate
- Max 13 tags

Respond ONLY with valid JSON.
"""
        parsed_output = await call_openai(prompt, image_b64=image_b64)

        try:
            output = ListingOutput(
                SKU=sku,
                Title=parsed_output.get("title",""),
                Description=parsed_output.get("description",""),
                Tags=merge_tags(short_keywords_for_tags, parsed_output.get("tags", []))
            )
        except ValidationError as e:
            logger.error(f"Validation error for SKU {sku}: {e}")
            return {"SKU": sku, "Title": "", "Description": "", "Tags": []}

        return output.dict()

    tasks = [process_listing(i, listing, images[i]) for i, listing in enumerate(listings)]
    results = await asyncio.gather(*tasks)

    csv_path = await save_csv_async(results)
    csv_id = str(uuid.uuid4())
    csv_store[csv_id] = csv_path

    return JSONResponse({"results": results, "csv_url": f"/download_csv/{csv_id}"})


# -------------------- BULK CSV ENDPOINT --------------------
@app.post("/generate_listings_csv")
async def generate_listings_csv(
    file: UploadFile = File(None),
    csv_file: UploadFile = File(None),
    upload: UploadFile = File(None),
    other: UploadFile = File(None)
):
    # Identify which file field has CSV
    csv_upload = next((f for f in [file, csv_file, upload, other] if f and f.filename), None)
    if not csv_upload:
        raise HTTPException(status_code=400, detail="No CSV uploaded.")

    content = await csv_upload.read()
    text = content.decode("utf-8", errors="ignore")

    # Parse CSV rows
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    listings = []
    for row in reader:
        sku = row[0].strip() if len(row) > 0 else f"row_{len(listings)}"
        image_url = row[1].strip() if len(row) > 1 else ""
        keywords = row[2].strip() if len(row) > 2 else ""
        notes = row[3].strip() if len(row) > 3 else ""
        listings.append({
            "sku": sku,
            "image_url": image_url,
            "keywords": keywords,
            "notes": notes
        })

    results = []

    async def process_csv_listing(i, listing):
        sku = listing.get("sku")
        raw_keywords = listing.get("keywords", "")
        all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
        optional_keywords_for_ai = [k for k in all_keywords if len(k)>TAG_MAX_LENGTH]
        short_keywords_for_tags = [k for k in all_keywords if len(k)<=TAG_MAX_LENGTH]

        prompt = f"""
You are an expert Etsy SEO copywriter.

Generate Etsy listing content in JSON format ONLY:
{{
  "title": "...",
  "description": "...",
  "tags": ["tag1","tag2",...,"tag20"]
}}

CRITICAL TITLE RULES:
- Title 45-85 chars
- Exclude long keywords: {json.dumps(optional_keywords_for_ai)}

DESCRIPTION RULES:
- At least 375 words
- First 300 chars must include all long keywords
- Notes: {listing.get('notes','')}

TAG RULES:
- 2-3 words per tag
- 10-20 chars
- Deduplicate
- Max 13 tags

Respond ONLY with valid JSON.
"""
        parsed_output = await call_openai(prompt, image_b64=listing.get("image_url"))
        try:
            output = ListingOutput(
                SKU=sku,
                Title=parsed_output.get("title",""),
                Description=parsed_output.get("description",""),
                Tags=merge_tags(short_keywords_for_tags, parsed_output.get("tags", []))
            )
        except ValidationError as e:
            logger.error(f"Validation error for CSV SKU {sku}: {e}")
            return {"SKU": sku, "Title": "", "Description": "", "Tags": []}
        return output.dict()

    tasks = [process_csv_listing(i, listing) for i, listing in enumerate(listings)]
    results = await asyncio.gather(*tasks)

    csv_path = await save_csv_async(results)
    csv_id = str(uuid.uuid4())
    csv_store[csv_id] = csv_path

    return JSONResponse({"results": results, "csv_url": f"/download_csv/{csv_id}"})


# -------------------- DOWNLOAD CSV --------------------
@app.get("/download_csv/{csv_id}")
async def download_csv(csv_id: str):
    csv_path = csv_store.get(csv_id)
    if csv_path and os.path.exists(csv_path):
        return FileResponse(csv_path, filename="filled_products.csv")
    return JSONResponse({"error":"CSV not found"}, status_code=404)


# -------------------- ROOT --------------------
@app.get("/")
def root():
    return {"message":"Etsy Listing Generator backend is running!"}




