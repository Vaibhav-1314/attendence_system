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
