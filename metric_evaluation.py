import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# This is the ground truth
coco_gt = COCO("custom.json")

# Load prediction list directly (e.g., from "annotations" inside a prediction dict)
with open("finetuned_pred.json", "r") as f:
    data = json.load(f)

# Extract the list of predictions
pred_list = data["annotations"]  # This must be a list of dicts

formatted_pred_list = []
for p in pred_list:
    # The category_id must align. If it is string, uncomment below
    # p["category_id"] = str(p["category_id"])
    formatted_pred_list.append(p)

coco_dt = coco_gt.loadRes(formatted_pred_list)

coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
# coco_eval.params.iouThrs = [
#     0.90
# ]  # set this to customize for e.g. [0.90] evaluate only for IoU 0.90
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
