# backend_fastapi.py
import os
import re
import time
import json
import tempfile
import pandas as pd
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

# -------------------- CONFIG --------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # use environment variable in Render
TAG_MAX_LENGTH = 20

# -------------------- HELPERS --------------------
def safe_parse_json(text: str):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return {}

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

class GenerateRequest(BaseModel):
    listings: List[ListingInput]
    examples: List[ExampleListing]
    shop_context: Optional[str] = ""
    shop_url: Optional[str] = ""

# -------------------- ENDPOINT --------------------
@app.post("/generate_listings")
async def generate_listings(
    request: GenerateRequest,
    images: List[UploadFile] = File(...),  # images uploaded in the same order as listings
):
    """
    Process multiple listings with images, keywords, and notes.
    Returns CSV file.
    """
    results = []
    temp_dir = tempfile.mkdtemp()

    for i, listing in enumerate(request.listings):
        try:
            # Save uploaded image
            image_file = images[i]
            image_path = os.path.join(temp_dir, image_file.filename)
            with open(image_path, "wb") as f:
                f.write(await image_file.read())

            # Separate keywords by length
            raw_keywords = listing.keywords or ""
            long_keywords_for_ai = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) > TAG_MAX_LENGTH]
            optional_keywords_str = ", ".join(long_keywords_for_ai)
            short_keywords_for_tags = [k.strip() for k in raw_keywords.split(",") if len(k.strip()) <= TAG_MAX_LENGTH]

            # Build examples string
            examples_str = ""
            for ex in request.examples:
                examples_str += f"""
Example:
{{
  "title": "{ex.title}",
  "description": "{ex.description}",
  "tags": {json.dumps(ex.tags)}
}}
"""

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

Rules:
- Product is a printable digital download.
- Use Optional Keywords exactly: {optional_keywords_str}
- Additional context: {listing.notes or ''}
- Shop context: {request.shop_context or ''}
- Shop URL: {request.shop_url or ''}
- Generate 20 tags max, deduplicate, tags <= {TAG_MAX_LENGTH} chars
- Respond ONLY with valid JSON
"""

            # Call OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                    ]
                }]
            )

            output_text = response.choices[0].message.content
            data = safe_parse_json(output_text)

            # Merge tags with short keywords
            tags = [t.strip() for t in data.get("tags", []) if len(t.strip()) <= TAG_MAX_LENGTH]
            all_tags = []
            for t in short_keywords_for_tags + tags:
                if t not in all_tags:
                    all_tags.append(t)

            # Build result row
            row = {
                "SKU": listing.sku,
                "Title": data.get("title", ""),
                "Description": data.get("description", ""),
                "Tags": ", ".join(all_tags[:13]),  # Etsy max 13 tags
            }
            results.append(row)
            print(f"✅ Processed SKU {listing.sku}")
            time.sleep(2)  # avoid rate limits

        except Exception as e:
            print(f"⚠️ Error processing {listing.sku}: {e}")
            results.append({
                "SKU": listing.sku,
                "Title": "",
                "Description": "",
                "Tags": "",
            })

    # Save results to CSV
    output_csv = os.path.join(temp_dir, "filled_products.csv")
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)

    return FileResponse(output_csv, filename="filled_products.csv")

# -------------------- RUN --------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
