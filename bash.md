kaggle competitions download -c bengaliai-cv19
mkdir data
unzip bengaliai-cv19.zip -d data

git init
git add . 
git commit

git remote add origin https://github.com/Dmitrsl/bengali
git pull origin master --allow-unrelated-histories

git push https://github.com/Dmitrsl/bengali


docker run ubuntu:20.04 -it -v /media/dmi/dmi_hard/kaggle/bengali:/home