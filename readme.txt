使い方

１、pythonコードと同じ階層にimagesフォルダを作成します。
２、適当に画像を集めます。(images/original/に保存する)
３、find_face.pyで顔だけを抜き出します。 (images/faces/に保存される)
４、クラスごとにフォルダを作り分類します。(images/faces/クラス名/画像 という階層)
５、make_datasets.py, learn.pyをそれぞれ実行します。
６、テスト画像を用意して、predict.pyを実行すると分類の結果が表示されます。


学習のところはgoogle colablatoryで実行するとうまくいきました。


windows10, python3.6.3

keras==2.3.1
tensorflow==2.2.0
opencv-python==4.2.0.34
numpy==1.18.4
selenium==3.141.0
chromedriver-binary==83.0.4103.39.0
requests==2.23.0