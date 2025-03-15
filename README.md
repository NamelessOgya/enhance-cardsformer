# cardsformerの再現実験    
## 誰向け  
cardsformerを手元で再現したい人      

## 概要   
  
## 実行準備  
### 1. ビルド済みコンテナの取得  
repositoryのルートディレクトリから
```
./make_new_sif_env.sh
```
これで./singularity/cafeobj.sifが作成される。  
  
(これは自分がローカルでビルドしたコンテナです。)  
(kagayaki上ではroot権限がなくビルドできないので...)
  
## インタラクティブ実行    
### 1. 作業ノードを起動し、singularityコンテナに入る    
以下を実行してインタラクティブインスタンスに入る。  
```
qsub -q -l select=1 -I
```
インスタンス内でrepositoryのルートディレクトリに移動し、
  
git clone
```
git clone https://github.com/WannianXia/Cardsformer.git ./tmp/cardsformer_clone
```  
  
コンテナへログイン  
```
./start_container.sh
```  

gitディレクトリへ移動  
```
cd ./tmp/cardsformer_clone
```  

## cloneからのバッチ実行　　
### 1. git clone  
略  
  
### 2. コンテナをダウンロード  
```
./make_new_sif_env.sh
```  
  
### 3. ./run.shの書き換え  
```runsh_generator.sh```を実行  
  
### 4. configファイルの配置  
```./config/config.ini```  

### 5.prediction modelの学習    
```qsub start_batch_job.sh```  
※prediction model, policy modelの移植も忘れずに。


