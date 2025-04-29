from dataset.nuclei_dab_util import *

# ihc_dir = "/root/Desktop/data/private/Dataset4Research/ER/TrainValAB/trainB"
# dab_save_dir = "/root/Desktop/data/private/Dataset4Research/ER/TrainValAB/train_IHC_dab"
# os.makedirs(dab_save_dir, exist_ok=True)
# mask_save_dir = "/root/Desktop/data/private/Dataset4Research/ER/TrainValAB/train_IHC_dab_mask"
# os.makedirs(mask_save_dir, exist_ok=True)

# # %%
# img_list = os.listdir(ihc_dir)
# img_list.sort()

# if_blur = True

# for i in range(len(img_list)):
#     time_s = time.time()
    
#     img_id = img_list[i]
#     ihc_bgr = cv2.imread(os.path.join(ihc_dir, img_id))
#     ihc_rgb = cv2.cvtColor(ihc_bgr, cv2.COLOR_BGR2RGB)
    
#     ihc_dab_rgb, ihc_dab_mask = get_dab_mask(ihc_rgb, if_blur)
#     ihc_dab_bgr = cv2.cvtColor(ihc_dab_rgb, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(os.path.join(dab_save_dir, img_id), ihc_dab_bgr)
#     cv2.imwrite(os.path.join(mask_save_dir, img_id), ihc_dab_mask)
    
#     time_e = time.time()
    
#     print("[{}/{} iter | time: {} s]---{} has been processed!".format(i+1,len(img_list),time_e-time_s,img_id))

ihc_dir = "/root/Desktop/data/private/Dataset4Research/ER/TrainValAB/trainB"
dab_mask_dir = "/root/Desktop/data/private/Dataset4Research/ER/TrainValAB/train_IHC_dab_mask"

# %%
map_save_dir = "/root/Desktop/data/private/Dataset4Research/ER/TrainValAB/train_IHC_nuclei_map"
os.makedirs(map_save_dir, exist_ok=True)
overlay_save_dir = "/root/Desktop/data/private/Dataset4Research/ER/TrainValAB/train_IHC_overlay"
os.makedirs(overlay_save_dir, exist_ok=True)

# %%
img_list = os.listdir(ihc_dir)
img_list.sort()

for i in range(len(img_list)):
    time_s = time.time()
    
    img_id = img_list[i]
    ihc_bgr = cv2.imread(os.path.join(ihc_dir, img_id))
    ihc_rgb = cv2.cvtColor(ihc_bgr, cv2.COLOR_BGR2RGB)
    ihc_dab_seg_mask = cv2.imread(os.path.join(dab_mask_dir, img_id), 0)
    
    ihc_h, ihc_h_rgb, ihc_dab, ihc_dab_rgb = get_ihc_channel(ihc_rgb)
    
    ihc_h_seg, ihc_h_mask, ihc_h_nuclei = get_h_nuclei(ihc_h)
    ihc_h_overlay = draw_nuclei(ihc_h_rgb, ihc_h_nuclei)
    
    ihc_dab_mask, ihc_dab_nuclei = get_dab_nuclei(ihc_dab_seg_mask)
    ihc_dab_overlay = draw_nuclei(ihc_dab_rgb, ihc_dab_nuclei)
    
    ihc_nuclei = ihc_h_nuclei + ihc_dab_nuclei
    ihc_overlay = draw_nuclei(ihc_rgb, ihc_nuclei)
    ihc_nuclei_map = get_nuclei_map(ihc_rgb, ihc_nuclei)
    
    cv2.imwrite(os.path.join(map_save_dir, img_id), ihc_nuclei_map)
    cv2.imwrite(os.path.join(overlay_save_dir, img_id), cv2.cvtColor(ihc_overlay, cv2.COLOR_RGB2BGR))
    
    time_e = time.time()
    
    print("[{}/{} iter | time: {} s]---{} has been processed!".format(i+1,len(img_list),time_e-time_s,img_id))
    