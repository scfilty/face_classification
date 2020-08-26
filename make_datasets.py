import os
import random
import shutil

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def make_datesets(src_dir, base_dir, train_number = 150, validation_number = 50, other_scale=4):

    class_names = os.listdir(src_dir)
    
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    dst_dirs = [train_dir, validation_dir, test_dir]
    
    
    for class_name in class_names:
        src_class_dir = os.path.join(src_dir, class_name)
        images = os.listdir(src_class_dir) #各要素はファイル名
        random.shuffle(images)

        #判別したいキャラとその他のキャラでデータ数が違うので調整
        scale = 1
        if class_name == "others":
            scale = other_scale

        images = [
            images[:train_number*scale],
            images[train_number*scale : train_number*scale + validation_number*scale],
            images[train_number*scale + validation_number*scale:]
        ]

        for i,dst_dir in enumerate(dst_dirs):
            make_dir(dst_dir)
            dst_class_dir = os.path.join(dst_dir, class_name)
            make_dir(dst_class_dir)

            for j,image in enumerate(images[i]):
                #画像のコピー
                #すでに存在してる時は上書きする
                src = os.path.join(src_class_dir, image)
                dst = os.path.join(dst_class_dir, f"{class_name}-{j}.png")
                shutil.copy(src, dst)


def main(now_dir):
    src_dir = os.path.join(now_dir, "images/original")
    dst_dir = os.path.join(now_dir, "learn_data")
    make_dir(dst_dir)
    make_datesets(src_dir, dst_dir)

if __name__ == "__main__":
    now_dir = os.getcwd()
    main(now_dir)
