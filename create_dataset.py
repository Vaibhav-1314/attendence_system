import cv2
import os
import shutil
import json

class_dictionary = {}

dataset = "dataset"

artifacts = "artifacts"

def capture(n):
    global class_dictionary

    cam = cv2.VideoCapture(0)

    dataset_folder = dataset

    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)
    os.mkdir(dataset_folder)

    for i in range(n):
        id = input("Enter your id: ")
        name = input("Enter your name: ")

        class_dictionary[id] = name

        while True:
            result, image = cam.read()

            if not result:
                print("Failed to capture")
                break

            cv2.imshow("Do Not Move!", image)

            k = cv2.waitKey(1)

            if k%256 == 27:
                print("Closing the app")
                break
            elif k%256 == 32:
                image_name = name + ".jpg"
                image_path = dataset_folder + "/" + image_name
                cv2.imwrite(image_path, image)
                print("image taken")

    cam.release()

if __name__ == "__main__":

    n = int(input("Enter no of students: "))

    capture(n)

    if os.path.exists(artifacts):
        shutil.rmtree(artifacts)
    os.mkdir(artifacts)
    
    class_dict_path = artifacts + "/" + "class_dictionary.json"
    with open(class_dict_path, 'w') as f:
        f.write(json.dumps(class_dictionary))
