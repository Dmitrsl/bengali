kaggle competitions download -c bengaliai-cv19
mkdir data
unzip bengaliai-cv19.zip -d data

git init
git add . 
git commit

git remote add origin https://github.com/Dmitrsl/bengali
git pull origin master --allow-unrelated-histories

git push https://github.com/Dmitrsl/bengali


docker run  -it -v /media/dmi/5F9CFB7847A8B8FE/kaggle/bengali:/home ubuntu:20.04
docker run -p 8888:8888 jupyter/datascience-notebook

