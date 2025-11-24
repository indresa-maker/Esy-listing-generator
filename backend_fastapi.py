# backend_fastapi_direct_images.py
import os
import re
import json
import tempfile
import asyncio
from typing import List
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

async def call_openai(prompt: str, images: List[UploadFile] = None):
    """Call OpenAI API (GPT-4o) with optional direct images"""
    messages = [{"role": "user", "content": prompt}]
    
    files_payload = []
    if images:
        for img in images:
            content = await img.read()
            files_payload.append({
                "name": img.filename,
                "type": img.content_type,
                "data": content
            })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        files=files_payload if files_payload else None
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

            # Build prompt
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
- Include 2 or 3 descriptive modifiers
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
            # Send image directly
            parsed_output = await call_openai(prompt, images=[images[i]])

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
