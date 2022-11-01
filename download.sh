mkdir demo
echo "Downloading $(tput setaf 1)Dataset for Demonstration$(tput sgr0) ..."

wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p27k77TWjknBYJ62EW97D2Xf_nElNZW3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p27k77TWjknBYJ62EW97D2Xf_nElNZW3" -O ./demo/Demo.zip && rm -rf ~/cookies.txt

echo "Downloading $(tput setaf 2)Model Checkpoints$(tput sgr0) ..."
wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AZr8eQa2m3fBkbb9t8MsWt-inbNwVVez' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AZr8eQa2m3fBkbb9t8MsWt-inbNwVVez" -O ./demo/checkpoints.zip && rm -rf ~/cookies.txt
echo "Download Complete."

cd demo
echo "Unzip files..."
unzip Demo.zip
mkdir ckpts
unzip checkpoints.zip -d ./ckpts
echo "Unzip Complete."
cd ..
