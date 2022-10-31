echo "Installing $(tput setaf 1)PyTorch$(tput sgr0) ..."
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
echo "Installing $(tput setaf 2)Matplotlib$(tput sgr0) ..."
conda install matplotlib==3.5.1 -y
echo "Installing $(tput setaf 3)imgaug$(tput sgr0) ..."
yes | pip install imgaug
echo "Installing $(tput setaf 2)utilities$(tput sgr0) ..."
conda install tqdm -y
conda install termcolor -y
conda install pandas -y
echo "Installing $(tput setaf 4)OpenCV$(tput sgr0) ..."
yes | pip install opencv-python==4.3.0.36
read -p "(Optional) Download $(tput setaf 1)VGG-16$(tput sgr0) for training or not: (y/n)?" choice
case "$choice" in
  y|Y ) wget https://download.pytorch.org/models/vgg16-397923af.pth -P ./loss/vgg/;;
  n|N ) echo "Cancel VGG-16 Downloading";;
  * ) echo "Invalid answer";;
esac
