import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('data/modalities.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        mesh_filename = item['mesh']
        rgb_filename = item['rgb']
        text_filename = item['text']
        file_id = text_filename.split('/')[-1][:-4]

        mesh_global= cv2.imread(mesh_filename)
        rgb_global = cv2.imread(rgb_filename)
        mesh_global = cv2.resize(mesh_global, (512, 512))
        rgb_global = cv2.resize(rgb_global, (512, 512))
        mesh_global = cv2.cvtColor(mesh_global, cv2.COLOR_BGR2RGB)
        rgb_global = cv2.cvtColor(rgb_global, cv2.COLOR_BGR2RGB)

        f = open(text_filename, 'r')
        text = f.readline()
        f.close()

        mesh_local, rgb_local = self.bounding_box(mesh_global, rgb_global)
        
        mesh_global = mesh_global.astype(np.float32) / 255.0
        mesh_local = mesh_local.astype(np.float32) / 255.0

        rgb_global = (rgb_global.astype(np.float32) / 127.5) - 1.0
        rgb_local = (rgb_local.astype(np.float32) / 127.5) - 1.0

        return dict(rgb_global=rgb_global, mesh_global=mesh_global, text=text, rgb_local=rgb_local, mesh_local=mesh_local, file_id=file_id)

    def bounding_box(self, mesh_global, rgb_global):
        mesh_global = cv2.cvtColor(mesh_global, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(mesh_global, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            bbox_0 = cv2.boundingRect(contours[0])
            bbox_1 = cv2.boundingRect(contours[1])
            (x, y, w, h) = self.merge_bounding_boxes(bbox_0, bbox_1)
        else:
            x, y, w, h = cv2.boundingRect(contours[0])
        mesh_global = cv2.cvtColor(mesh_global, cv2.COLOR_GRAY2RGB)
        
        mesh_local = mesh_global[y:y+h, x:x+w]
        rgb_local = rgb_global[y:y+h, x:x+w]
        mesh_local = cv2.resize(mesh_local, (512, 512))
        rgb_local = cv2.resize(rgb_local, (512, 512))
        
        return mesh_local, rgb_local
    
    def merge_bounding_boxes(self, bbox1, bbox2):
        x_min = min(bbox1[0], bbox2[0])
        y_min = min(bbox1[1], bbox2[1])
        x_max = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y_max = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        new_x = x_min
        new_y = y_min
        new_w = x_max - x_min
        new_h = y_max - y_min
        
        merged_bbox = (new_x, new_y, new_w, new_h)
        return merged_bbox
