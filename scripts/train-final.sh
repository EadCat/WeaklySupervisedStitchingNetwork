python train.py \
--name 01-final \
--total-epochs 20 \
--iter-op-freq 100 \
--iter-save-freq -1 \
--iter-sample-freq -1 \
--sample-num 1 \
--loss PL1 SSIM \
--appearance-weight 0. \
--div-appear False \
--ssim-weight 0.4 \
--div-ssim True \
--perceptual-weight 0.6 \
--div-perceptual False \
--vgg-loss-weight 0. 0. 0.2 0.3 0.5 \
--optim Adam \
--lr 1e-4 \
--unet large \
--reg large \
--homography 1 \
--generator double \
--local-adj-limit 0.3 \
--strict True \
--smart True \
--gpu 0 1 2 3 \
--npgpu 4 \
--world-size 4 \
--dataroot /home/user/SSD/ECCV-Split/ \
--train-datalist /home/user/SSD/ECCV-Split/erp_train_triple_woc.txt \
--batch-size 3 \
--num-workers 10 \
--shuffle True \
--pin-memory True \
--transform resize normalize augment \
--resize 512 1024 \
--mean 0.5 0.5 0.5 \
--std 0.5 0.5 0.5 \
--aug-prob 0.5
