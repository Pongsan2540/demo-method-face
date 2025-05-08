from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
import shutil
import os
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import requests
from requests.auth import HTTPBasicAuth
import json
import urllib.request
import cv2
import numpy as np

from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_DIR = "uploads"


'''
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.chmod(UPLOAD_DIR, 0o777)

folder_1 = "image_output"
os.makedirs(folder_1, exist_ok=True)
'''

# Jinja2 setup
templates = Environment(loader=FileSystemLoader("templates"))

# Serve images
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/image_output", StaticFiles(directory="image_output"), name="image_output")

@app.get("/", response_class=HTMLResponse)
async def index():

    images = [f for f in os.listdir(UPLOAD_DIR) if f.endswith((".jpg", ".jpeg", ".png"))]
    template = templates.get_template("index.html")
    return HTMLResponse(content=template.render(images=images), status_code=200)


class Person(BaseModel):
    name: str
    age: int
    gender: str
    profession: str


@app.get("/test/", response_class=JSONResponse)
async def test():
    try:

        def search_data_by_id(url_database, name_database, doc_id):
            try:
                es_url = url_database + "/" + name_database + "/_doc/" + doc_id
                
                auth = ("elastic", "changeme")  # ใส่ username และ password ของ Elasticsearch
                headers = {"Content-Type": "application/json"}

                # Perform the search request to get the document by _id
                response = requests.get(es_url, auth=auth, headers=headers)

                #print(response.status_code)  
                if response.status_code == 200:
                    #print(response.json())  # Output the document content
                    #print(json.dumps(response.json(), indent=4))
                    result = response.json()
                else:
                    result = f"Error: {response.status_code}"

            except Exception as e:
                #print(f"Error: {e}")
                result = f"Error: {e}"

            return result

        def extract_object_color(image, bbox):
            x1, y1, x2, y2 = map(int, bbox)  # แปลงค่า bounding box เป็นตัวเลข
            roi = image[y1:y2, x1:x2]  # Crop ภาพใน bounding box

            return roi

        url = "http://localhost:9200/my-vertor/_search"
        payload = {
            "size": 1000,  
            "query": {
                "match_all": {} 
            }
        }

        response = requests.get(url, json=payload, auth=HTTPBasicAuth('elastic', 'changeme'))
        link_imges_ref = None


        # Check the response status and print the formatted result
        if response.status_code == 200:
            # Extracting only the "id_database" from each hit
            hits = response.json().get('hits', {}).get('hits', [])
            data_list = []
            counter = 1  
            
            for hit in hits[:]:   

                id_database = hit['_source'].get('id_database')

                url_database = "http://0.0.0.0:9200"
                name_database = "my-databaes-general"

                result = search_data_by_id(url_database, name_database, id_database)
                link_imges = result['_source'].get('urlImage')
                location = result['_source'].get('location')
                timeCaptureImage = result['_source'].get('timeCaptureImage')
                timeStamp = result['_source'].get('timeStamp')

                nameCam = result['_source'].get('nameCam')
                typeCam = result['_source'].get('typeCam')
                gps = result['_source'].get('gps')
                map = result['_source'].get('map')

                entry = {
                    "id": counter,  # ใช้ตัวเลขลำดับเป็น id
                    "details": {
                        "id_database":  hit['_source'].get('id_database'),
                        "text":  hit['_source'].get('text'),
                        "bbox": hit['_source'].get('bbox'),
                        "time_stamp": hit['_source'].get('time_stamp'),
                        "link_imges": link_imges,
                        "location": location,
                        "timeCaptureImage": timeCaptureImage,
                        "timeStamp": timeStamp,
                        "nameCam": nameCam,
                        "typeCam": typeCam,
                        "gps": gps,
                        "label": hit['_source'].get('label'),
                        "map": map
                    }
                }

                data_list.append(entry)
                counter += 1  # เพิ่มตัวนับ

            return JSONResponse(content=data_list, status_code=200)
        else:
            print(f"Error: {response.status_code}")


    except:
        pass

class TextInput(BaseModel):
    text: str

@app.post("/send_text/", response_class=JSONResponse)
async def send_text(input: TextInput):


    def search_data_by_id(url_database, name_database, doc_id):
        try:
            es_url = url_database + "/" + name_database + "/_doc/" + doc_id
            
            auth = ("elastic", "changeme")  # ใส่ username และ password ของ Elasticsearch
            headers = {"Content-Type": "application/json"}

            # Perform the search request to get the document by _id
            response = requests.get(es_url, auth=auth, headers=headers)

            #print(response.status_code)  
            if response.status_code == 200:
                #print(response.json())  # Output the document content
                #print(json.dumps(response.json(), indent=4))
                result = response.json()
            else:
                result = f"Error: {response.status_code}"

        except Exception as e:
            #print(f"Error: {e}")
            result = f"Error: {e}"

        return result


    #input_text = input.text  # Get the text from the input
    input_text = GoogleTranslator(source="th", target="en").translate(input.text)

    vector = model_SentenceTransformer.encode(str(input_text))

    url = "http://localhost:9200/my-vertor/_search"
    headers = {'Content-Type': 'application/json'}

    data = {
        "size": 3,
        "query": {
            "knn": {
                "field": "vector",
                "query_vector": vector.tolist(),  # Convert vector to list
                "k": 3
            }
        }
    }

    response = requests.post(url, headers=headers, json=data, auth=('elastic', 'changeme'))
    hits = response.json().get('hits', {}).get('hits', [])

    if response.status_code == 200:

        hits = response.json().get('hits', {}).get('hits', [])
        data_list = []
        counter = 1  

        for hit in hits[:]:   
                
            id_database = hit['_source'].get('id_database')

            url_database = "http://0.0.0.0:9200"
            name_database = "my-databaes-general"

            result = search_data_by_id(url_database, name_database, id_database)

            link_imges = result['_source'].get('urlImage')
            location = result['_source'].get('location')
            timeCaptureImage = result['_source'].get('timeCaptureImage')
            timeStamp = result['_source'].get('timeStamp')

            nameCam = result['_source'].get('nameCam')
            typeCam = result['_source'].get('typeCam')
            gps = result['_source'].get('gps')
            map = result['_source'].get('map')

            entry = {
                "id": counter,  # ใช้ตัวเลขลำดับเป็น id
                "details": {
                    "id_database":  hit['_source'].get('id_database'),
                    "text":  hit['_source'].get('text'),
                    "bbox": hit['_source'].get('bbox'),
                    "time_stamp": hit['_source'].get('time_stamp'),
                    "link_imges": link_imges,
                    "location": location,
                    "timeCaptureImage": timeCaptureImage,
                    "timeStamp": timeStamp,
                    "nameCam": nameCam,
                    "typeCam": typeCam,
                    "gps": gps,
                    "label": hit['_source'].get('label'),
                    "map": map
                }
            }

            data_list.append(entry)
            counter += 1  # เพิ่มตัวนับ

        #print(json.dumps(hits, indent=4, ensure_ascii=False))
        return JSONResponse(content=data_list, status_code=200)

if __name__ == "__main__":

    model_SentenceTransformer = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    uvicorn.run(app, host="192.168.110.21", port=5325, log_level="info")
