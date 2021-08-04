import shutil
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
import json

import numpy as np
import cv2

app = FastAPI()


def list_to_string(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1


@app.post("/match_template")
async def match_template(image: UploadFile = File(...), x_token=Header(...)):
    with open("../credentials/cred.json", 'r') as kf:
        credentials = json.load(kf)
    if credentials["API_KEY"] != x_token:
        raise HTTPException(401, detail="Unauthorized access to the API")
    else:
        pass
    with open("../inputs/input.png", "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    micr_number = []

    with open("../cmc7-templates.json", 'r') as f:
        temps = json.load(f)
    images = ['three', 'zero', 'one', 'five', 'four', 'two', 'nine', 'eight', 'seven', 'six']
    coordinates = []
    detections = []
    pattern = []
    output = {
        "coordinates": coordinates,
        "pattern": pattern
    }
    for i in images:
        temp_path = temps['characters'][i]['img']
        image = cv2.imread("../inputs/input.png")
        (h, w,) = image.shape[:2]
        delta = int(h - (h * 0.2))
        bottom = image[delta:h, 0:w]
        img_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        template = cv2.imread("../" + temp_path, 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = temps['characters'][i]['threshold']
        loc = np.where(res >= threshold)
        num_count = 0

        mask = np.zeros(bottom.shape[:2], np.uint8)
        for pt in zip(*loc[::-1]):
            if mask[pt[1] + int(round(h / 2)), pt[0] + int(round(w / 2))] != 255:
                mask[pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 255
                num_count += 1
                match = {
                    "TOP_LEFT_X": pt[0],
                    "TOP_LEFT_Y": pt[1],
                    "BOTTOM_RIGHT_X": pt[0] + w,
                    "BOTTOM_RIGHT_Y": pt[1] + h,
                    "MATCH_VALUE": res[pt[1], pt[0]],
                    "LABEL": temps['characters'][i]['ref'],
                }

                detections.append(match)
                coordinates.append(match["TOP_LEFT_X"])

                for detection in detections:
                    rect = cv2.rectangle(bottom, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)
                    label = cv2.putText(
                        rect,
                        f"{temps['characters'][i]['ref']}",
                        (pt[0] + w + 2, pt[1] + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

    for c in sorted(coordinates):
        for d in detections:
            if d["TOP_LEFT_X"] == c:
                # print(d["LABEL"])
                micr_number.append(d["LABEL"])

    return list_to_string(micr_number)


@app.get("/")
async def get_main():
    return "Success"
