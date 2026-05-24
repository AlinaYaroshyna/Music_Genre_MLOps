from fastapi import FastAPI
from fastapi import UploadFile
import shutil
import os
from app.feature_extractor import FeatureExtractor
from app.predictor import GenrePredictor

app = FastAPI()
extractor = FeatureExtractor()
predictor = GenrePredictor()

@app.get("/")
def home():
    return {
        "message": ("Music Genre Classification API")
    }


@app.post("/predict")
async def predict(file: UploadFile):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(
        temp_dir,
        file.filename
    )

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted = extractor.extract_all(temp_path)
    results = predictor.predict(extracted)
    os.remove(temp_path)

    return results