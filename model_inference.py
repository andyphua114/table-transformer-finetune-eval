import json
import os
import warnings

import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

warnings.filterwarnings("ignore", category=UserWarning)


def inference(type, folder_path, model_path, image_dict):
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    annotations = {}
    annotations["annotations"] = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    model = TableTransformerForObjectDetection.from_pretrained(model_path).to(device)

    for idx, image_path in enumerate(files):
        image = [Image.open(image_path)]

        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        target_sizes = torch.tensor([img.size[::-1] for img in image])
        results = image_processor.post_process_object_detection(
            outputs, threshold=0.01, target_sizes=target_sizes
        )

        for result in results:
            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                b = [i for i in box.tolist()]
                boxes = [b[0], b[1], b[2] - b[0], b[3] - b[1]]
                a_dict = {}
                a_dict["image_id"] = image_dict[image_path.replace("\\", "/")]
                if type == "detection":
                    a_dict["category_id"] = 0  # for detection
                elif type == "structure":
                    a_dict["category_id"] = str(label.tolist())  # for structure
                a_dict["bbox"] = boxes
                a_dict["score"] = round(score.item(), 3)
                annotations["annotations"].append(a_dict)

        if idx % 500 == 0:
            print("Completed: {}".format(idx))
    return annotations


if __name__ == "__main__":
    # custom.json is the post processed exported COCO-format labeled data from Label Studio
    with open("custom.json", "r") as file:
        data = json.load(file)

    image_dict = {}
    for d in data["images"]:
        filename = d["file_name"]
        image_dict[filename] = d["id"]

    # use either "detection" or "structure"
    type = "detection"

    if type == "detection":
        # for detection
        folder_path = r"C:\Users\andyp\projects\label-studio\data"
        model_path = r"microsoft/table-transformer-detection"  # change to your fine-tuned model path
    elif type == "structure":
        # for structure
        folder_path = r"C:\Users\andyp\projects\label-studio\data-table-images"
        model_path = r"microsoft/table-transformer-structure-recognition"  # change to your fine-tuned model path
    else:
        raise ValueError()

    annotations = inference(type, folder_path, model_path, image_dict)

    with open("finetuned_pred.json", "w") as f:
        json.dump(annotations, f, indent=2)
