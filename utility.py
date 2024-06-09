import numpy as np
import math

def LBP(img, r, p):
    '''
    
     References:
     ----------
        [1] T. Ojala, M. Pietikainen, T. Maenpaa, "Multiresolution gray-scale
           and rotation invariant texture classification with local binary
           patterns", IEEE Transactions on Pattern Analysis and Machine
           Intelligence, vol. 24, no. 7, pp. 971-987, July 2002
           https://doi.org/10.1109/TPAMI.2002.1017623
    
    -> Extracts the neighbours and then extracts the pattern 
    -> based on the pattern bitstring, shift it into an int
    -> set the int to that pixel indcies in LBP_IMAGE
    
    '''    
    img_pad = np.pad(img, pad_width=r, mode='constant', constant_values=0)
    lbp_img = np.zeros(shape=img.shape, dtype="uint8")
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):    
            nbr = extract_neighbourhood(img_pad, i, j, r)
            patt_arr = extract_pattern(nbr, img[i,j], p)
            n = 0
            for b in patt_arr:
                if b == 1:
                    n = n << 1
                    n = n + 1
                    # n = n * 2
                if b == 0:
                    n = n << 1
            lbp_img[i,j] = n
    return lbp_img


def extract_neighbourhood(I, x, y, r, debug=False):
    '''
    Extract neighbours at radius r for I[x,y]
    The extract neighbours are 4 array of all values on the boundary starting from top left and moving in clockwise direction
    '''    
    px = x + r
    py = y + r
    
    dim = 1 + r * 2
    
    chunk = I[px - r: px + r + 1, py - r: py + r + 1]
    chunk_transpose = chunk.T

    if debug:
        print(chunk)
        print(I[x,y])
    
    row_1 = chunk[0]
    row_2 = chunk.T[dim - 1]
    row_3 =  chunk[dim - 1][::-1]
    row_4 = chunk.T[0][::-1 ]
    
    return [row_1, np.delete(row_2, 0), np.delete(row_3, 0), np.delete(row_4, [0,(row_4.shape[0] - 1)])] #clickwise


def extract_pattern(neighbourhood, pixel, p):
    '''    
    Extract pattern from neighbourhood array for sampling values p, for example neighbourhood can have 24 values and p=8,
    so we will make bins of 3 and pick middle value 

     STEPS:
     -----
     1) combine neighbourhood to 1D array
     2) for each window of p values, pick center value if middle index is integer
     3) if middle index is not integer, perform bilinear interpolation between last and next pixel values
     4) Finally, check condition 
         if  pixel_value >= I(xi,xj) {
            set value to 1
         }
         else {
            set value to 0
         }
     5) Flatten and return
    '''    
    pattern = []
    
    for i in range(4):
        row_patt = neighbourhood[i].tolist()
        pattern = pattern + row_patt 

    final_arr = []
    start = 0
    r = len(pattern) / p
    
    for j in range(p):
        end = start + r + 1
        mid = (start + end) / 2

        if isinstance(mid, int):
            final_arr.append(pattern[mid - 1])
        else:
            upper = math.ceil(mid)
            lower = math.floor(mid)
            avg = (pattern[upper - 1] + pattern[lower - 1]) // 2            
            final_arr.append(avg)

        start = end - 1

    pattern = np.where(pixel >= np.array(final_arr) , 1, 0)        
    return pattern.flatten()

