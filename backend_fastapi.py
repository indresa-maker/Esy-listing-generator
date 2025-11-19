# backend_fastapi.py
import os
import re
import json
import time
import tempfile
import pandas as pd
import asyncio
from typing import List, Optional
import requests

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
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
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON:", text)
        return {}
    return {}

def generate_prompt(listing, examples, shop_context, shop_url, optional_keywords_str):
    """Build prompt for OpenAI"""
    examples_str = ""
    for ex in examples:
        examples_str += f"""
Example:
{{
  "title": "{ex['title']}",
  "description": "{ex['description']}",
  "tags": {json.dumps(ex['tags'])}
}}
"""
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

Rules:
- Product is a printable digital download.
- Use Optional Keywords exactly: {optional_keywords_str}
- Additional context: {listing.get('notes', '')}
- Shop context: {shop_context}
- Shop URL: {shop_url}
- Generate 20 tags max, deduplicate, tags <= {TAG_MAX_LENGTH} chars
- Respond ONLY with valid JSON
"""
    return prompt

async def call_openai(prompt, image_url=None):
    """Call OpenAI API with optional image URL"""
    content = [{"type": "text", "text": prompt}]
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}]
    )
    raw_text = response.choices[0].message.content
    print("🟢 OpenAI raw response:", raw_text)
    return safe_parse_json(raw_text)

def build_csv(results):
    temp_dir = tempfile.mkdtemp()
    output_csv = os.path.join(temp_dir, "filled_products.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    return output_csv

def upload_to_fileio(file_path):
    """Upload a file to file.io and return temporary public URL"""
    with open(file_path, "rb") as f:
        resp = requests.post("https://file.io", files={"file": f}, data={"expires":"1d"})
    print("📤 file.io raw response:", resp.text)
    if resp.status_code != 200:
        raise Exception(f"file.io upload failed: {resp.status_code} {resp.text}")
    data = resp.json()
    if not data.get("success"):
        raise Exception(f"file.io upload failed: {data}")
    return data["link"]

# -------------------- FASTAPI SETUP --------------------
app = FastAPI(title="Etsy Listing Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- DATA MODELS --------------------
class ExampleListing(BaseModel):
    title: str
    description: str
    tags: List[str]

# -------------------- ENDPOINT 1: App / FormData --------------------
@app.post("/generate_listings_app")
async def generate_listings_app(
    request: str = Form(...),
    images: List[UploadFile] = File(...),
):
    try:
        data = json.loads(request)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse request JSON:", request)
        return {"error": "Invalid JSON in request", "details": str(e)}

    listings = data.get("listings", [])
    examples = data.get("examples", [])
    shop_context = data.get("shop_context", "")
    shop_url = data.get("shop_url", "")

    results = []

    for i, listing in enumerate(listings):
        sku = listing.get("sku", f"row_{i}")
        try:
            # Save image temporarily
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
            raw_keywords = listing.get("keywords", "")
            long_keywords_for_ai = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) > TAG_MAX_LENGTH]
            optional_keywords_str = ", ".join(long_keywords_for_ai)
            short_keywords_for_tags = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) <= TAG_MAX_LENGTH]

            # Prompt & OpenAI
            prompt = generate_prompt(listing, examples, shop_context, shop_url, optional_keywords_str)
            print(f"📝 Prompt for SKU {sku}:\n{prompt[:500]}...")  # first 500 chars

            try:
                parsed_output = await call_openai(prompt, image_url=image_url)
            except Exception as e:
                print(f"⚠️ OpenAI call failed for SKU {sku}: {e}")
                parsed_output = {}

            # Merge tags
            tags = [t.strip() for t in parsed_output.get("tags", []) if len(t.strip()) <= TAG_MAX_LENGTH]
            all_tags = []
            for t in short_keywords_for_tags + tags:
                if t not in all_tags:
                    all_tags.append(t)

            results.append({
                "SKU": sku,
                "Title": parsed_output.get("title", ""),
                "Description": parsed_output.get("description", ""),
                "Tags": ", ".join(all_tags[:13])
            })
            print(f"✅ Processed SKU {sku}")
            await asyncio.sleep(2)

        except Exception as e:
            print(f"⚠️ Error processing SKU {sku}: {e}")
            results.append({
                "SKU": sku,
                "Title": "",
                "Description": "",
                "Tags": ""
            })

    output_csv = build_csv(results)
    print(f"✅ CSV generated at: {output_csv}")
    return FileResponse(output_csv, filename="filled_products.csv")

# -------------------- ENDPOINT 2: CSV upload --------------------
@app.post("/generate_listings_csv")
async def generate_listings_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    results = []

    for _, row in df.iterrows():
        try:
            listing = {
                "sku": row.get("sku",""),
                "keywords": row.get("keywords",""),
                "notes": row.get("notes","")
            }
            examples = []  # optionally pull from row or elsewhere
            shop_context = row.get("shop_context","")
            shop_url = row.get("shop_url","")

            raw_keywords = listing.get("keywords", "")
            long_keywords_for_ai = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) > TAG_MAX_LENGTH]
            optional_keywords_str = ", ".join(long_keywords_for_ai)
            short_keywords_for_tags = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) <= TAG_MAX_LENGTH]

            prompt = generate_prompt(listing, examples, shop_context, shop_url, optional_keywords_str)

            image_url = row.get("image_url","")  # must be public URL
            parsed_output = await call_openai(prompt, image_url=image_url if image_url else None)

            tags = [t.strip() for t in parsed_output.get("tags", []) if len(t.strip()) <= TAG_MAX_LENGTH]
            all_tags = []
            for t in short_keywords_for_tags + tags:
                if t not in all_tags:
                    all_tags.append(t)

            results.append({
                "SKU": listing["sku"],
                "Title": parsed_output.get("title",""),
                "Description": parsed_output.get("description",""),
                "Tags": ", ".join(all_tags[:13])
            })
            print(f"✅ Processed SKU {listing['sku']}")
            await asyncio.sleep(2)

        except Exception as e:
            print(f"⚠️ Error processing row {row.get('sku','')}: {e}")
            results.append({
                "SKU": row.get("sku",""),
                "Title": "",
                "Description": "",
                "Tags": ""
            })

    output_csv = build_csv(results)
    print(f"✅ CSV generated at: {output_csv}")
    return FileResponse(output_csv, filename="filled_products.csv")

# -------------------- OPTIONAL ROOT --------------------
@app.get("/")
def root():
    return {"message": "Etsy Listing Generator backend is running!"}

# -------------------- RUN --------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
