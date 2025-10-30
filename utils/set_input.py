import numpy as np

def set_input_plrr_inpainting(img_degraded, mask):
    height, width, n_channel = img_degraded.shape[0], img_degraded.shape[1], img_degraded.shape[-1]
    # mask = self.mask # [hh,ww,in_c]
    # img_degraded = img_degraded.copy()
    
    valid_loc = np.sum(mask,axis=-1) == n_channel
    semivalid_loc = np.logical_and( (~ valid_loc) , np.sum(mask,axis=-1)!=0 )
    
    # process semivalid values
    denom = np.sum(mask, -1)
    denom[denom==0] = 1
    semivalid_value = np.sum(img_degraded,-1) / denom
    semivalid_value = np.stack([semivalid_value]*n_channel, -1)
    
    
    semivalid_loc = np.stack([semivalid_loc]*n_channel, -1)==1
    semivalid_loc = np.logical_and(semivalid_loc , np.logical_not(mask) )
    img_degraded[semivalid_loc] = semivalid_value[semivalid_loc]

    # process invalid values
    init_invalid = (~ valid_loc) & (np.sum(mask,-1)==0)
    invalid = init_invalid
    process_invalid = 1
    win_size = 3
    count = 0
    while process_invalid:
        index,indey = np.nonzero(invalid==1)
        for k in range(len(index)):
            i = index[k]
            j = indey[k]
            x1 = min(max(i-win_size,0),height-1)
            x2 = min(max(i+win_size,0),height-1)
            y1 = min(max(j-win_size,0),width-1)
            y2 = min(max(j+win_size,0),width-1)
            # local_area = img_degraded[x1:x2, y1:y2, :]
            local_area = img_degraded[x1:x2, y1:y2, ...]
            local_area = np.reshape(local_area, [local_area.shape[0]*local_area.shape[1], *local_area.shape[2:]])
            # img_degraded[i,j,:]=np.median(local_area,0)
            img_degraded[i,j,...]=np.median(local_area,0)

        invalid = np.logical_and(np.sum(img_degraded,-1)<=0.012 , init_invalid)
        count = count +1
        if np.sum(invalid)==0 or count>=50:
            process_invalid = 0
        else:
            process_invalid = 1
        if (count+1)%10==0:
            win_size = win_size+1
    
    return img_degraded