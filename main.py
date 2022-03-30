import os
import face_recognition as fr
import cv2
import numpy as np

# dataset_path = "./small_dataset/"
# name = os.listdir(dataset_path)
#
# folder = "./Dataset/test/"
# for data in range(len(name)):
#     sub = os.listdir(dataset_path + name[data] + "/")
#     for i in range(len(sub)):
#         dst = f"{name[data]}{str(i)}.jpeg"
#         src = f"{dataset_path}{name[data]}/{sub[i]}"
#         dst = f"{folder}{dst}"
#         os.rename(src, dst)

path = "./Dataset/train/"
face_name = []
face_encoding = []

images = os.listdir(path)
for _ in images:
    image = fr.load_image_file(path + _)
    image_path = path + _
    encoding = fr.face_encodings(image)[0]

    face_encoding.append(encoding)
    face_name.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

print(face_name)

test_dir = "./Dataset/test/"
output_dir = "./Output/"
test_images = os.listdir(test_dir)
x= test_dir+test_images[0]
for test in range(len(test_images)):
    image = cv2.imread(test_dir+test_images[test])
    face_locations = fr.face_locations(image)
    face_encodings = fr.face_encodings(image, face_locations)

    for (top, right, bottom, left), face in zip(face_locations, face_encodings):
        matches = fr.compare_faces(face_encoding, face)
        name = ""

        face_distances = fr.face_distance(face_encoding, face)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = face_name[best_match]

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(image, (left, bottom - 15), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imwrite(output_dir + str(test)+".jpeg", image)


