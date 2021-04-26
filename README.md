# Walmart
Liaro課題提出用リポジトリ

## requirements
* Docker 20.10.5
* docker-compose 1.25.0
* NVIDIA Docker 2.5.0

### build enviroment 
<pre>
docker-compose up -d
</pre>

## directory tree
<pre>
.
├── input                     <---- input
├── output                    <---- モデルの予測、feature importance等を入れるディレクトリ
├── report                    <---- レポートが入ったディレクトリ(容量が100M越えてしまったので、出力は消去済)
├── eda                       <---- eda用ファイルが入ったディレクトリ(容量が100M越えてしまったので、出力は消去済)
├── src                    
|   ├──base_model.ipynb       <---- 学習、検証、推論を行うファイル
|
|__ module                    <---- モデルや特徴量を作るモジュールが置かれたディレクトリ
</pre>

## setup
<pre>
#データのダウンロード
cd input/
kaggle competitions download -c walmart-recruiting-store-sales-forecasting
yes | unzip '*.zip'
yes | unzip '*.zip'
</pre>
