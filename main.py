import imp
from fastapi import FastAPI
from detect_people.detect_people import DetectPeople

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/detect_people")
def detect_people():
    detect_people = DetectPeople()
    return {"detect_people": detect_people.detect_people()}
