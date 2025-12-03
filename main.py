# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import listings, ads_analyzer

app = FastAPI(title="Etsy Listing Generator API")

# ---------------- CORS Configuration ----------------
origins = [
    "https://c08091be-56b2-4932-bf23-98ce122b41e1.lovableproject.com",  # frontend domain
    "http://localhost:5173",  # local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Routers ----------------
app.include_router(listings.router, prefix="/listings")
app.include_router(ads_analyzer.router, prefix="/ads")
