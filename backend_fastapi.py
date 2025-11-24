# backend_fastapi.py
import os
import re
import json
import tempfile
import asyncio
from typing import List
import requests
from fastapi import FastAPI, File, UploadFile, Form
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
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise Exception(f"file.io upload failed: {data}")
    return data["link"]

async def call_openai(prompt, image_url=None):
    """Call OpenAI API (GPT-4o) and return parsed JSON"""
    messages = [{"role": "user", "content": prompt}]
    if image_url:
        messages.append({"role": "user", "content": f"[IMAGE] {image_url}"})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    raw_text = response.choices[0].message.content
    print("🟢 OpenAI raw response:", raw_text[:500])
    return safe_parse_json(raw_text)

def build_csv(results):
    import csv, uuid
    
    filename = f"/tmp/{uuid.uuid4()}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["SKU", "Title", "Description", "Tags"])
        for item in results:
            tags_list = item.get("Tags", [])
            writer.writerow([
                item.get("SKU", ""),
                item.get("Title", ""),
                item.get("Description", ""),
                ", ".join(tags_list)
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

csv_store = {}

# -------------------- SINGLE LISTING ENDPOINT --------------------
@app.post("/generate_listings_app")
async def generate_listings_app(
    request: str = Form(...),
    images: List[UploadFile] = File(...)
):
    try:
        data = json.loads(request)
    except Exception as e:
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
            except Exception as e:
                print(f"⚠️ file.io upload failed for SKU {sku}: {e}")
                image_url = None

            # Keywords
            raw_keywords = listing.get("keywords","")
            all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
            optional_keywords_for_ai = [k for k in all_keywords if len(k) > TAG_MAX_LENGTH]
            short_keywords_for_tags = [k for k in all_keywords if len(k) <= TAG_MAX_LENGTH]

            # Build examples string
            examples_str = ""
            for ex in examples:
                title = json.dumps(ex.get("title", ""))
                description = json.dumps(ex.get("description", ""))
                tags = json.dumps(ex.get("tags", []))
                examples_str += f"Example:\n{{\n  \"title\": {title},\n  \"description\": {description},\n  \"tags\": {tags}\n}}\n\n"

            # -------- Your original prompt incorporated --------
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
- Include all long keywords (>20 chars): {json.dumps(optional_keywords_for_ai)}
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
            parsed_output = await call_openai(prompt, image_url=image_url)

            # Merge tags
            tags = [t.strip() for t in parsed_output.get("tags", []) if len(t.strip()) <= TAG_MAX_LENGTH]
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

            await asyncio.sleep(2)  # avoid rate limits

        except Exception as e:
            print(f"⚠️ Error processing SKU {sku}: {e}")
            results.append({"SKU": sku,"Title":"","Description":"","Tags":[]})

    # Save CSV
    csv_path = build_csv(results)
    csv_id = str(len(csv_store)+1)
    csv_store[csv_id] = csv_path

    return JSONResponse({
        "results": results,
        "csv_url": f"/download_csv/{csv_id}"
    })

# -------------------- CSV UPLOAD ENDPOINT --------------------
@app.post("/generate_listings_csv")
async def generate_listings_csv(
    file: UploadFile = File(None),
    csv: UploadFile = File(None),
    upload: UploadFile = File(None),
    other: UploadFile = File(None)
):
    # Pick the first non-empty file
    csv_file = next((f for f in [file, csv, upload, other] if f and f.filename), None)
    if not csv_file:
        return JSONResponse({"error": "No CSV file uploaded."}, status_code=400)

    content = await csv_file.read()
    text = content.decode("utf-8", errors="ignore")

    import csv as csv_lib, io
    reader = csv_lib.reader(io.StringIO(text))
    header = next(reader, None)

    listings = []
    for row in reader:
        sku = row[0].strip() if len(row) > 0 else f"row_{len(listings)}"
        image_url = row[1].strip() if len(row) > 1 else ""
        keywords = row[2].strip() if len(row) > 2 else ""
        notes = row[3].strip() if len(row) > 3 else ""
        listings.append({"sku": sku, "image_url": image_url, "keywords": keywords, "notes": notes})

    # Process CSV exactly like single listings
    results = []
    examples = []  # or fetch from somewhere if needed
    shop_context = ""
    shop_url = ""

    for listing in listings:
        sku = listing.get("sku")
        try:
            raw_keywords = listing.get("keywords","")
            all_keywords = [k.strip() for k in raw_keywords.split(",") if k.strip()]
            optional_keywords = [k for k in all_keywords if len(k) > TAG_MAX_LENGTH]
            short_keywords = [k for k in all_keywords if len(k) <= TAG_MAX_LENGTH]

            # Examples string empty if none
            examples_str = ""
            for ex in examples:
                title = json.dumps(ex.get("title", ""))
                description = json.dumps(ex.get("description", ""))
                tags = json.dumps(ex.get("tags", []))
                examples_str += f"Example:\n{{\n  \"title\": {title},\n  \"description\": {description},\n  \"tags\": {tags}\n}}\n\n"

            # Build prompt using your original logic
            prompt = f"..."  
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
- Include all long keywords (>20 chars): {json.dumps(optional_keywords_for_ai)}
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

        except Exception as e:
            print(f"⚠️ Error processing CSV SKU {sku}: {e}")
            results.append({"SKU": sku,"Title":"","Description":"","Tags":[]})

    csv_path = build_csv(results)
    csv_id = str(len(csv_store)+1)
    csv_store[csv_id] = csv_path

    return JSONResponse({"results": results,"csv_url": f"/download_csv/{csv_id}"})

# -------------------- DOWNLOAD CSV --------------------
@app.get("/download_csv/{csv_id}")
def download_csv(csv_id: str):
    csv_path = csv_store.get(csv_id)
    if csv_path and os.path.exists(csv_path):
        return FileResponse(csv_path, filename="filled_products.csv")
    return JSONResponse({"error":"CSV not found"}, status_code=404)

# -------------------- ROOT --------------------
@app.get("/")
def root():
    return {"message":"Etsy Listing Generator backend is running!"}
