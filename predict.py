import os, cv2, glob, time
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


def predict(characters, image_path, model_name = "9-nine-.h5", target_size = (150,150)):
    model = load_model(model_name)

    classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    original_img = cv2.imread(image_path)
    faces = classifier.detectMultiScale(original_img)
    if len(faces) == 0:
        print("can't find face")
        return 0

    for x,y,w,h in faces:
        face_image = original_img[y:y+h, x:x+w]
        #cv2.imwrite("face.png",face_image)
        time.sleep(0.5)
        img = image.load_img("face.png", target_size=target_size)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        pred = model.predict(img_tensor, batch_size=1, verbose=0)[0]

        pred_index = np.argmax(pred)
        text = characters[pred_index] + str(round(pred[pred_index],2))

        cv2.rectangle(original_img, (x,y), (x+w,y+h), color=(0,0,255), thickness=3)
        cv2.putText(original_img, text, (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    
    cv2.imwrite('faces.png',original_img)


def main():
    characters = ["haruka", "miyako", "noa", "others", "sora"]
    predict(characters, "test.png")

if __name__ == "__main__":
    main()