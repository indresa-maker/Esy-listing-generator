# backend_fastapi.py
import os
import re
import json
import time
import base64
import tempfile
import pandas as pd
import asyncio
from typing import List, Optional

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
    """Parse JSON safely from AI output"""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {}

def generate_prompt(listing, examples, shop_context, shop_url, optional_keywords_str):
    """Build the AI prompt"""
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

async def call_openai(prompt, image_path=None):
    """Call OpenAI API with optional base64 image"""
    content = [{"type": "text", "text": prompt}]
    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        content.append({"type": "image_data", "image_data": img_b64})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}]
    )
    return safe_parse_json(response.choices[0].message.content)

def build_csv(results):
    """Save results to a temporary CSV file and return path"""
    temp_dir = tempfile.mkdtemp()
    output_csv = os.path.join(temp_dir, "filled_products.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    return output_csv

# -------------------- FASTAPI SETUP --------------------
app = FastAPI(title="Etsy Listing Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- DATA MODELS --------------------
class ExampleListing(BaseModel):
    title: str
    description: str
    tags: List[str]

class ListingInput(BaseModel):
    sku: str
    keywords: Optional[str] = ""
    notes: Optional[str] = ""

# -------------------- ENDPOINT 1: App / FormData --------------------
@app.post("/generate_listings_app")
async def generate_listings_app(
    request: str = Form(...),
    images: List[UploadFile] = File(...),
):
    data = json.loads(request)
    listings = data["listings"]
    examples = data.get("examples", [])
    shop_context = data.get("shop_context", "")
    shop_url = data.get("shop_url", "")

    results = []

    for i, listing in enumerate(listings):
        try:
            # Save image and encode as base64
            image_file = images[i]
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, image_file.filename)
            with open(image_path, "wb") as f:
                f.write(await image_file.read())

            # Keywords
            raw_keywords = listing.get("keywords", "")
            long_keywords_for_ai = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) > TAG_MAX_LENGTH]
            optional_keywords_str = ", ".join(long_keywords_for_ai)
            short_keywords_for_tags = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) <= TAG_MAX_LENGTH]

            # Prompt & OpenAI
            prompt = generate_prompt(listing, examples, shop_context, shop_url, optional_keywords_str)
            parsed_output = await call_openai(prompt, image_path=image_path)

            # Merge tags
            tags = [t.strip() for t in parsed_output.get("tags", []) if len(t.strip()) <= TAG_MAX_LENGTH]
            all_tags = []
            for t in short_keywords_for_tags + tags:
                if t not in all_tags:
                    all_tags.append(t)

            results.append({
                "SKU": listing["sku"],
                "Title": parsed_output.get("title", ""),
                "Description": parsed_output.get("description", ""),
                "Tags": ", ".join(all_tags[:13])
            })
            print(f"✅ Processed SKU {listing['sku']}")
            await asyncio.sleep(2)

        except Exception as e:
            print(f"⚠️ Error processing {listing.get('sku','')}: {e}")
            results.append({
                "SKU": listing.get("sku",""),
                "Title": "",
                "Description": "",
                "Tags": ""
            })

    output_csv = build_csv(results)
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

            # Use image_url from CSV if provided
            image_url = row.get("image_url","")  # must be public URL
            parsed_output = await call_openai(prompt, image_path=None if not image_url else None)  # TODO: handle public URL differently

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
