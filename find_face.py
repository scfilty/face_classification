import os
import cv2
import unicodedata

def is_japanese(string):
    for char in string:
        name = unicodedata.name(char) 
        if "CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return True
    return False


def find_face():
    classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    now_dir = os.getcwd()
    src_dir = os.path.join(now_dir, "images/original")
    dir_names = os.listdir(src_dir)

    for dir_name in dir_names:
        if is_japanese(dir_name):
            raise Exception("フォルダ名をアルファベットに変更してください")

        image_dir = os.path.join(src_dir, dir_name)
        image_pathes = list(map(lambda x:os.path.join(image_dir,x), os.listdir(image_dir)))

        for i, image_path in enumerate(image_pathes):
            
            image = cv2.imread(image_path)
            faces = classifier.detectMultiScale(image)

            output_base_dir = os.path.join(now_dir, "images/faces")
            output_dir = os.path.join(output_base_dir, dir_name)
            if not os.path.exists(output_base_dir): os.makedirs(output_base_dir)
            if not os.path.exists(output_dir): os.makedirs(output_dir)

            for j, (x,y,w,h) in enumerate(faces):
                face_image = image[y:y+h, x:x+w]
                image_path = os.path.join(output_dir, f"{i}-{j}.png")
                cv2.imwrite(image_path,face_image)


def main():
    find_face()

if __name__ == "__main__":
    main()