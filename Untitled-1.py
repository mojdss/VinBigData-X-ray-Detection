
from ultralytics.data.annotator import auto_annotate

auto_annotate(data=r"d:\paper code\Non Carious", det_model="yolo11x.pt", sam_model="sam2_b.pt")