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
            print(f"📝 Prompt for SKU {sku}:\n{prompt[:500]}...")  # print first 500 chars

            try:
                parsed_output = await call_openai(prompt, image_url=image_url)
                print(f"🟢 OpenAI raw response for SKU {sku}: {parsed_output}")
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
            await asyncio.sleep(2)  # avoid rate limits

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
