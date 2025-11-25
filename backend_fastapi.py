# backend_fastapi.py
import os
import re
import json
import time
import tempfile
import pandas as pd
import asyncio
from typing import List
import requests
import csv, io

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# -------------------- CONFIG --------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
TAG_MAX_LENGTH = 20

# -------------------- HELPERS --------------------
def safe_parse_json(text: str):
    """Safely parse JSON from OpenAI output"""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print("❌ Failed to parse JSON:", text, e)
    return {}

def upload_to_fileio(file_path):
    """Upload a file to file.io and return temporary public URL"""
    with open(file_path, "rb") as f:
        resp = requests.post("https://file.io", files={"file": f}, data={"expires":"1d"})
    print("📤 file.io response:", resp.text)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise Exception(f"file.io upload failed: {data}")
    return data["link"]

async def call_openai(prompt, image_url=None):
    """Call OpenAI API and return parsed JSON"""
    content = [{"type":"text","text":prompt}]
    if image_url:
        content.append({"type":"image_url","image_url":{"url":image_url}})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":content}]
    )
    raw_text = response.choices[0].message.content
    print("🟢 OpenAI raw response:", raw_text)
    return safe_parse_json(raw_text)

def build_csv(results):
    import csv, uuid, os
    
    filename = f"/tmp/{uuid.uuid4()}.csv"
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["SKU", "Title", "Description", "Tags"])
        
        for item in results:
            # Convert tags list → comma-separated string
            tags_list = item.get("Tags", [])
            tag_string = ", ".join(tags_list)

            writer.writerow([
                item.get("SKU", ""),
                item.get("Title", ""),
                item.get("Description", ""),
                tag_string  # 👈 now clean
            ])
    
    return filename

# -------------------- FASTAPI SETUP --------------------
app = FastAPI(title="Etsy Listing Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary store of CSVs
csv_store = {}

# -------------------- ENDPOINT 1 --------------------
@app.post("/generate_listings_app")
async def generate_listings_app(
    request: str = Form(...),
    images: List[UploadFile] = File(...)
):
    # Parse JSON from form
    try:
        data = json.loads(request)
    except Exception as e:
        print("❌ Failed to parse request JSON:", request, e)
        return JSONResponse({"error":"Invalid JSON in request", "details": str(e)}, status_code=400)

    listings = data.get("listings", [])
    examples = data.get("examples", [])
    shop_context = data.get("shop_context", "")
    shop_url = data.get("shop_url", "")

    results = []

    for i, listing in enumerate(listings):
        sku = listing.get("sku", f"row_{i}")
        try:
            # Save uploaded image
            image_file = images[i]
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, image_file.filename)
            with open(image_path, "wb") as f:
                f.write(await image_file.read())

            # Upload to file.io
            try:
                image_url = upload_to_fileio(image_path)
                print(f"✅ Uploaded image for SKU {sku}: {image_url}")
            except Exception as e:
                print(f"⚠️ file.io upload failed for SKU {sku}: {e}")
                image_url = None

            # Keywords
            raw_keywords = listing.get("keywords","")
            all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
            optional_keywords_for_ai = [k for k in all_keywords if len(k)>TAG_MAX_LENGTH]
            short_keywords_for_tags = [k for k in all_keywords if len(k)<=TAG_MAX_LENGTH]
            optional_keywords_str = json.dumps(optional_keywords_for_ai)

            # Build examples string
            examples_str = ""
            for ex in examples:
                examples_str += f"""
Example:
{{
  "title": "{ex.get('title','')}",
  "description": "{ex.get('description','')}",
  "tags": {json.dumps(ex.get('tags',[]))}
}}
"""

            # Build prompt
            prompt = f"""
You are an expert Etsy SEO copywriter.

Here are examples of Etsy listings to follow:
{examples_str}

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
- Title MUST be between 45 and 85 characters
- Title MUST clearly identify the product type
- Include 2–3 descriptive modifiers
- DO NOT repeat words
- DO NOT use any long keywords (>20 characters)
- DO NOT include every keyword; keep it natural, descriptive but short

DESCRIPTION RULES:
- Description must be at least 375 words
- First 300 characters MUST be extremely keyword rich
- Use line breaks, bullet points, numbered lists
- Include all long keywords (>20 chars): {json.dumps(optional_keywords_str)}
- Long keywords must NEVER appear in the title or tags
- Clearly identify product type, style, colors, use cases, room placement, and features
- Notes: {listing.get('notes','')}
- Shop context: {shop_context}
- Shop URL: {shop_url}

TAG RULES:
- Tags MUST be 2–3 words each
- Each tag MUST be 10–20 characters
- Deduplicate tags
- Generate up to 20 tags
- Avoid generic tags, make them specific and long-tail

IMPORTANT:
Respond ONLY with valid JSON.
"""
            print(f"📝 Prompt for SKU {sku}:\n{prompt[:500]}...")  # first 500 chars

            parsed_output = await call_openai(prompt, image_url=image_url)

            # Merge tags
            tags = [t.strip() for t in parsed_output.get("tags",[]) if len(t.strip())<=TAG_MAX_LENGTH]
            all_tags = []
            for t in short_keywords_for_tags + tags:
                if t not in all_tags:
                    all_tags.append(t)

            results.append({
                "SKU": sku,
                "Title": parsed_output.get("title",""),
                "Description": parsed_output.get("description",""),
                "Tags": all_tags[:13]  # Etsy max 13
            })
            print(f"✅ Processed SKU {sku}")
            await asyncio.sleep(2)  # avoid rate limits

        except Exception as e:
            print(f"⚠️ Error processing SKU {sku}: {e}")
            results.append({"SKU": sku,"Title":"","Description":"","Tags":[]})

    # Save CSV
    csv_path = build_csv(results)
    csv_id = str(len(csv_store)+1)
    csv_store[csv_id] = csv_path

    # Return JSON with results + CSV URL
    return JSONResponse({
        "results": results,
        "csv_url": f"/download_csv/{csv_id}"
    })

# -------------------- BULK CSV ENDPOINT --------------------
@app.post("/generate_listings_csv")
async def generate_listings_csv(
    file: UploadFile = File(None),   # Lovable uses "file"
    csv: UploadFile = File(None),    # fallback
    upload: UploadFile = File(None), # fallback
    other: UploadFile = File(None)   # fallback for unknown keys
):
    """
    CSV upload endpoint with fixed column positions:
    A: SKU
    B: Image URL (optional)
    C: Optional keywords
    D: Optional notes
    """

    # Pick the first non-empty file
    csv_file = next((f for f in [file, csv, upload, other] if f and f.filename), None)
    if not csv_file:
        return JSONResponse(
            {"error": "No CSV file uploaded. Make sure you send form-data with a file."},
            status_code=400
        )

    print(f"📄 Received CSV file: {csv_file.filename}")

    # Read CSV
    content = await csv_file.read()
    text = content.decode("utf-8", errors="ignore")

    import csv as csv_lib, io
    reader = csv_lib.reader(io.StringIO(text))
    header = next(reader, None)  # skip header row if present

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

    print(f"📦 Loaded {len(listings)} listings from CSV")

    # ---- continue with your existing logic below ----

    examples = []
    shop_context = ""
    shop_url = ""
    results = []

    examples_str = ""
    for ex in examples:
        examples_str += f"""
Example:
{{
  "title": "{ex.get('title','')}",
  "description": "{ex.get('description','')}",
  "tags": {json.dumps(ex.get('tags',[]))}
}}
"""

    for i, listing in enumerate(listings):
        sku = listing.get("sku")

        try:
            raw_keywords = listing.get("keywords", "")
            all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
            optional_keywords = [k for k in all_keywords if len(k) > TAG_MAX_LENGTH]
            short_keywords = [k for k in all_keywords if len(k) <= TAG_MAX_LENGTH]

            prompt = f"""
You are an expert Etsy SEO copywriter.

Here are examples of Etsy listings to follow:
{examples_str}

Generate Etsy listing content in JSON format ONLY:
{{
  "title": "...",
  "description": "...",
  "tags": ["tag1","tag2",...,"tag20"]
}}

CRITICAL TITLE RULES:
- Title MUST be between 45 and 85 characters
- Title MUST clearly identify the product type
- Include 2–3 descriptive modifiers
- DO NOT repeat words
- DO NOT use any long keywords (>20 characters)
- DO NOT include every keyword; keep it natural, descriptive but short

DESCRIPTION RULES:
- Description must be at least 375 words
- First 300 characters MUST be extremely keyword rich
- Use line breaks, bullet points, numbered lists
- Include all long keywords (>20 chars): {json.dumps(optional_keywords)}
- Long keywords must NEVER appear in the title or tags
- Clearly identify product type, style, colors, use cases, room placement, and features
- Notes: {listing.get('notes','')}
- Shop context: {shop_context}
- Shop URL: {shop_url}

TAG RULES:
- Tags MUST be 2–3 words each
- Each tag MUST be 10–20 characters
- Deduplicate tags
- Generate up to 20 tags
- Avoid generic tags, make them specific and long-tail

IMPORTANT:
Respond ONLY with valid JSON.
"""

            parsed = await call_openai(prompt, image_url=listing.get("image_url"))

            tags = [t.strip() for t in parsed.get("tags", []) if len(t.strip()) <= TAG_MAX_LENGTH]

            merged_tags = []
            for t in short_keywords + tags:
                if t not in merged_tags:
                    merged_tags.append(t)

            results.append({
                "SKU": sku,
                "Title": parsed.get("title", ""),
                "Description": parsed.get("description", ""),
                "Tags": merged_tags[:13]
            })

            print(f"✅ Processed CSV SKU {sku}")

        except Exception as e:
            print(f"⚠️ Error processing CSV SKU {sku}: {e}")
            results.append({
                "SKU": sku,
                "Title": "",
                "Description": "",
                "Tags": []
            })

    # Save CSV
    csv_path = build_csv(results)
    csv_id = str(len(csv_store)+1)
    csv_store[csv_id] = csv_path

    return JSONResponse({
        "results": results,
        "csv_url": f"/download_csv/{csv_id}"
    })
          
# Endpoint to download CSV
@app.get("/download_csv/{csv_id}")
def download_csv(csv_id: str):
    csv_path = csv_store.get(csv_id)
    if csv_path and os.path.exists(csv_path):
        return FileResponse(csv_path, filename="filled_products.csv")
    return JSONResponse({"error":"CSV not found"}, status_code=404)

# Optional root endpoint
@app.get("/")
def root():
    return {"message":"Etsy Listing Generator backend is running!"}

# Run server locally
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)








