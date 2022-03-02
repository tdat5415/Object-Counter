
#######################################################################
import numpy as np

def corn2xywh(boxes):
    xy = (boxes[..., :2] + boxes[..., 2:]) / 2.0
    wh = boxes[..., 2:] - boxes[..., :2]
    # return tf.concat([xy,wh], axis=-1)
    return np.concatenate([xy,wh], axis=-1)

def xywh2corn(boxes):
    tl = boxes[..., :2] - boxes[..., 2:] / 2.0
    br = boxes[..., :2] + boxes[..., 2:] / 2.0
    # return tf.concat([xymin, xymax], axis=-1)
    return np.concatenate([tl, br], axis=-1)

def corn2tlwh(boxes):
    tl = boxes[..., :2]
    wh = boxes[..., 2:] - boxes[..., :2]
    return np.concatenate([t1, wh], axis=-1)

def tlwh2corn(boxes):
    tl = boxes[..., :2]
    br = boxes[..., :2] + boxes[..., 2:]
    return np.concatenate([tl, br], axis=-1)

def xywh2tlwh(boxes):
    tl = boxes[..., :2] - boxes[..., 2:] / 2.0
    wh = boxes[..., 2:]
    return np.concatenate([tl, wh], axis=-1)

def tlwh2xywh(boxes):
    xy = boxes[..., :2] + boxes[..., 2:] / 2.0
    wh = boxes[..., 2:]
    return np.concatenate([xy, wh], axis=-1)

#######################################################################
import numpy as np

def centroid_tracking(centroids_t0, centroids_t1): # shape[n, 2], shape[m, 2]
    if len(centroids_t0) == 0 or len(centroids_t1) == 0: return []
    t0_ids = [i for i in range(len(centroids_t0))]
    dist_2d = []
    for cen_t0 in centroids_t0:
        cen_t0 = cen_t0[None, :]
        dist_row = np.linalg.norm(cen_t0-centroids_t1, axis=-1)
        dist_2d.append(dist_row)
    
    dist_2d_row_wise_sort = sorted(zip(dist_2d, t0_ids), key=lambda x:min(x[0]))

    pairs = []
    selected_t1 = set()
    for dist_t0_to_t1, t0_id in dist_2d_row_wise_sort:
        t1_id = np.argmin(dist_t0_to_t1)
        if t1_id in selected_t1: continue
        else: selected_t1.add(t1_id)
        pairs.append((t0_id, t1_id))
        
    return pairs

def tracking_id(before_ids, pairs, after_boxes_len, cnt_t0): # ex) before_ids = [100,101,102,103,104,105]
    # current_id = max(before_ids)
    pairs = sorted(pairs, key=lambda x:x[1], reverse=True)
    after_ids = []
    for i in range(after_boxes_len):
        if pairs and i == pairs[-1][1]:
            idx0, _ = pairs.pop()
            after_ids.append(before_ids[idx0])
        else:
            # current_id += 1
            cnt_t0 += 1
            after_ids.append(cnt_t0)
    cnt_t1 = cnt_t0
    return after_ids, cnt_t1

#######################################################################
import numpy as np

def light_shade(img, alpha=1.5, beta=1.0):
    img = (img - beta*128)*alpha + beta*128
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

#######################################################################
import json
import numpy as np

def json2label(path): # json 경로
    with open(path, "r") as f:
        data = json.load(f)
    labels = [shape["label"] for shape in data["shapes"]]
    boxes = [shape["points"] for shape in data["shapes"]]
    boxes = np.array([box[0] + box[1] for box in boxes])
    # boxes = corn2xywh(boxes)
    imgsize = (data["imageWidth"], data["imageHeight"])
    return labels, boxes, imgsize # rank1, rank2, rank1
  
def normalize_boxes(imgsize, boxes, mode=''): # xy, 0~512 -> 0~1
    imgsize = np.concatenate([imgsize, imgsize])
    if mode=="reverse":
        boxes *= imgsize
    else:
        boxes /= imgsize
    return boxes # rank2
  
def make_text_label(ids, boxes, path): # 숫자, 0~1boxes, 파일이름
    assert len(ids) == len(boxes)
    with open(path, 'w', encoding="utf-8") as f:
        for id, box in zip(ids, boxes):
            f.write("{} {} {} {} {}\n".format(id, *box))

#######################################################################
import numpy as np
import cv2
# xywh2corn, normalize_boxes

def read_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        data = [list(map(float, row.rstrip().split())) for row in rows]
    return np.array(data)


def show_annotation(img, groupby_dic):
    for key in groupby_dic:
        seed = sum(map(lambda x:ord(x), key))
        ids = groupby_dic[key]['ids']
        boxes = groupby_dic[key]['boxes']
        box_names = list(map(lambda x:key + str(x), ids))
        np.random.seed(seed)
        color = tuple(map(int, np.random.randint(0, 255, size=3)))
        img = draw_boxes(img, box_names, boxes, color, xywh=False)
    return img

def draw_boxes(img, names, boxes, color=None, xywh=True):
    boxes = np.array(boxes)
    if xywh:
        boxes = xywh2corn(boxes)
    boxes = boxes.astype(np.int32)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, box in zip(names, boxes):
        if color is None:
            color = tuple(map(int, np.random.randint(0, 255, size=3)))
        img = cv2.rectangle(img, box[:2], box[2:], color, 3)
        cv2.putText(img, str(name), box[:2], font, 1, color, 2, cv2.LINE_AA)
    return img

def show_annotation2(img_path, txt_path):
    img = cv2.imread(img_path)
    data = read_txt(txt_path)
    img_size = img.shape[:2][::-1]
    names, boxes = data[:,0], data[:,1:]
    boxes = normalize_boxes(img_size, boxes, mode='reverse')
    new_img = draw_boxes(img, names, boxes)
    cv2.imshow('test', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#######################################################################
import cv2
import random as rd
from tqdm import tqdm
import numpy as np
from glob import glob
import os
# json2label, make_text_label, normalize_boxes, corn2xywh

DATASET_DIR_PATH = "./dataset"
SAVE_IMAGES_DIR_PATH = os.path.join(DATASET_DIR_PATH, "images")
SAVE_LABELS_DIR_PATH = os.path.join(DATASET_DIR_PATH, "labels")
BG_IMG_DIR_PATH = "./seed_data/conveyor_img"
OBG_IMG_DIR_PATH = "./seed_data/product_imgs"
OBG_JSON_DIR_PATH = "./seed_data/product_json"

def random_resize(product_img, img_size, boxes): # 0.7 ~ 1.5
    alpha = 0.8*rd.random() + 0.7
    new_size = np.array(img_size) * alpha
    new_size = new_size.astype(np.int32)
    new_boxes = boxes * alpha
    product_img = product_img.astype(np.uint8)
    new_img = cv2.resize(product_img, new_size)
    return new_img, new_size, new_boxes

def random_point(bg_size, img_size):
    range_x = bg_size[0] - img_size[0]
    range_y = bg_size[1] - img_size[1]
    return rd.randint(0, range_x), rd.randint(0, range_y)

def synthetic_img(product_img, bg_img, img_size, r_xy):
    r_x, r_y = r_xy
    x, y = img_size
    assert r_x+x <= bg_img.shape[1] and r_y+y <= bg_img.shape[0], "합성범위 넘어감"
    bg_img[r_y:r_y+y, r_x:r_x+x] = product_img
    return bg_img

def gen_data_one(product_img, product_label, bg_img, NAME_TO_ID): # return img, shape(n,5)
    names, boxes, img_size = product_label # 제품과 날짜 box
    assert product_img.shape[0] < bg_img.shape[0]
    assert product_img.shape[1] < bg_img.shape[1]
    assert product_img.shape[:2][::-1] == img_size, f"{product_img.shape[::-1]} {img_size}"
    
    bg_size = bg_img.shape[:2][::-1]
    product_img, img_size, boxes = random_resize(product_img, img_size, boxes) # 제품이미지와 박스 사이즈 변경
    r_xy = random_point(bg_size, img_size) # 배경이미지안의 랜덤좌표
    assert type(r_xy) == tuple
    new_img = synthetic_img(product_img, bg_img, img_size, r_xy) # 이미지합성
    boxes += r_xy*2 # (a,b,a,b) # 이동한 만큼 박스 좌표도 변경

    ids = np.array([NAME_TO_ID[name] for name in names])[:,None]
    boxes = normalize_boxes(bg_size, boxes) # ndarray
    boxes = corn2xywh(boxes)
    assert len(ids)==len(boxes)
    label = np.concatenate([ids, boxes], axis=-1)
    return new_img, label
    
def gen_data(NAME_TO_ID, num=10):
    bg_img_path = glob(BG_IMG_DIR_PATH + "/*.jpg")[0]
    bg_img0 = cv2.imread(bg_img_path)
    product_img_paths = sorted(glob(OBG_IMG_DIR_PATH + "/*.jpg"))
    product_json_paths = sorted(glob(OBG_JSON_DIR_PATH + "/*.json"))
    product_imgs = [cv2.imread(path) for path in product_img_paths]
    product_labels = [json2label(path) for path in product_json_paths] # xywh
    for i in tqdm(range(num)):
        v = rd.choice(range(len(product_imgs)))
        bg_img = bg_img0.copy()
        img, label = gen_data_one(product_imgs[v], product_labels[v], bg_img, NAME_TO_ID) # img, shape(n, 5) # 0 0.123 0.234 0.345 0.456
        cv2.imwrite(SAVE_IMAGES_DIR_PATH + "/" + f"{i:04d}.jpg", img)
        make_text_label(label[:,0], label[:, 1:], SAVE_LABELS_DIR_PATH + "/" + f"{i:04d}.txt")
        
#######################################################################
import socket as sk

def connection_for_server():
    HOST = ''
    PORT = 8888
    server = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    with_client, addr = server.accept()
    print('Server : Connected.')
    print(addr)
    return with_client

def connection_for_client():
    HOST = '127.0.0.1'
    PORT = 8888
    client = sk.socket(sk.AF_INET, sk.SOCK_STREAM)
    client.connect((HOST, PORT))
    print('Client : Connected.')
    return client
        
#######################################################################
from collections import defaultdict
import numpy as np
import cv2

def im_rotate(img, degree):
    h, w = img.shape[:-1]
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))
    return result

def auto_rotation(img, barcode_img):
    rad_counter = defaultdict(int)
    edges = cv2.Canny(img, 50, 200)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 130)
    for line in lines:
        r, theta = line[0]
        rad_counter[theta] += 1
    if len(lines) == 0: return None
    rad = max(rad_counter, key=lambda x:rad_counter[x])
    deg = rad/np.pi*180
    new_img = im_rotate(img, deg-180)
    return new_img

#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
