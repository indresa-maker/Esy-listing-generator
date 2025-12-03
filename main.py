from fastapi import FastAPI
from routers import listings, ads_analyzer

app = FastAPI(title="Etsy Tools Backend")

# Include routers
app.include_router(listings.router, prefix="/listings")
app.include_router(ads_analyzer.router, prefix="/ads")

@app.get("/")
def root():
    return {"message": "Etsy Tools backend is running!"}

# Optional local run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
