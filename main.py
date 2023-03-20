from fastapi import FastAPI
from detect_people.detect_people import DetectPeople
from dotenv import load_dotenv
from requests import post
from os import getenv

load_dotenv()

app = FastAPI()


@app.on_event("startup")
def startup():
    try:
        post(f"{getenv('backend')}/awake", json={"id": getenv('id')})
    except:
        pass
    global dp
    dp = DetectPeople()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/detect_people")
def detect_people():
    return {"detect_people": dp.detect_people()}


@app.on_event("shutdown")
def shutdown():
    try:
        global dp
        del dp
        post(f"{getenv('backend')}/awake/dead", json={"id": getenv('id')})
    except:
        print("Error")
        pass
