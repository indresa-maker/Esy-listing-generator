# ads_analyzer.py
import base64
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
import pandas as pd

router = APIRouter()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------- Helper Functions ----------------
def image_to_base64(file: UploadFile) -> str:
    content = file.file.read()
    return base64.b64encode(content).decode("utf-8")

def analyze_ads_data(df: pd.DataFrame):
    df = df.copy()
    numeric_cols = ['views','clicks','click_rate','orders','revenue','spend','ROAS']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['CVR'] = df.apply(lambda row: (row['orders']/row['clicks']*100) if row['clicks']>0 else 0, axis=1)
    
    results = []
    for _, row in df.iterrows():
        rec, commentary = "", ""
        if row['ROAS'] >= 2 and row['click_rate'] >= 1 and row['CVR'] >= 3 and row['revenue'] > row['spend']:
            rec = "KEEP"
            commentary = "Profitable and performing well."
        elif row['ROAS'] < 1 or row['click_rate'] < 0.5 or (row['orders']==0 and row['spend']>10):
            rec = "PAUSE"
            commentary = "Underperforming or wasting budget."
        else:
            rec = "OPTIMIZE"
            if row['click_rate'] < 1 and row['CVR'] >=3:
                commentary = "Improve title or thumbnail to increase CTR."
            elif row['click_rate'] >=1 and row['CVR'] <2:
                commentary = "Optimize listing page or description to improve CVR."
            else:
                commentary = "Moderate performance; review ad for tweaks."

        results.append({
            "Title": row['title'],
            "CTR": round(row['click_rate'],2),
            "CVR": round(row['CVR'],2),
            "ROAS": round(row['ROAS'],2),
            "Spend": round(row['spend'],2),
            "Revenue": round(row['revenue'],2),
            "Recommendation": rec,
            "Commentary": commentary
        })
    return results

# ---------------- Preflight Endpoint ----------------
@router.options("/analyze_screenshot", responses={200: {"description": "Preflight OK"}})
async def ads_preflight():
    return JSONResponse(status_code=200, content={"message": "ok"})
    
# ---------------- Endpoint ----------------
@router.post("/analyze_screenshot")
async def analyze_etsy_ads_screenshot(
    file: UploadFile = None,
    image_base64: str = Form(None),
    strategy: str = Form("balanced")
):
    try:
        if file:
            image_b64 = image_to_base64(file)
        elif image_base64:
            image_b64 = image_base64
        else:
            return JSONResponse({"error": "No image provided"}, status_code=400)

        # GPT step
        system_prompt = """
You are an AI assistant specialized in reading Etsy Ads dashboard screenshots.
Extract all visible ad rows into a structured table with columns:
listing, title, views, clicks, click_rate, orders, revenue, spend, ROAS.
Return JSON array of ads in the same order as they appear in the screenshot.
If numbers are unclear, estimate reasonably and indicate in notes.
Respond ONLY in valid JSON.
"""
        user_prompt = f"Etsy Ads screenshot in base64:\n{image_b64}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}"}
                ]}
            ],
        )

        import json
        try:
            ad_data = json.loads(response.choices[0].message.content)
        except Exception:
            return JSONResponse({"error": "Failed to parse GPT output", "raw_output": response.choices[0].message.content}, status_code=500)

        df = pd.DataFrame(ad_data)
        analyzed_results = analyze_ads_data(df)
        return JSONResponse({"results": analyzed_results})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
