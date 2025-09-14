import logging
import os
import warnings
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
    Trainer,
    TrainingArguments,
)
from transformers.image_transforms import center_to_corners_format

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(
            img_folder, "custom_train.json" if train else "custom_val.json"
        )
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return pixel_values, target


def collate_fn(batch):
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection"
    )
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


def train():
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection"
    )

    # change the img_folder to the directory where you have the training/eval images
    train_dataset = CocoDetection(
        img_folder=r"C:\Users\andyp\projects\label-studio\data",
        processor=processor,
        train=True,
    )

    val_dataset = CocoDetection(
        img_folder=r"C:\Users\andyp\projects\label-studio\data",
        processor=processor,
        train=False,
    )

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    def convert_bbox_yolo_to_pascal(boxes, image_size):
        """
        Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
        to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

        Args:
            boxes (torch.Tensor): Bounding boxes in YOLO format
            image_size (Tuple[int, int]): Image size in format (height, width)

        Returns:
            torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
        """
        # convert center to corners format
        boxes = center_to_corners_format(boxes)

        # convert to absolute coordinates
        height, width = image_size
        boxes = boxes * torch.tensor([[width, height, width, height]])

        return boxes

    id2label = {0: "table"}
    label2id = {"table": 0}

    @dataclass
    class ModelOutput:
        logits: torch.Tensor
        pred_boxes: torch.Tensor

    @torch.no_grad()
    def compute_metrics(
        evaluation_results, image_processor, threshold=0.0, id2label=None
    ):
        """
        Compute mean average mAP, mAR and their variants for the object detection task.

        Args:
            evaluation_results (EvalPrediction): Predictions and targets from evaluation.
            threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
            id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

        Returns:
            Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
        """

        predictions, targets = (
            evaluation_results.predictions,
            evaluation_results.label_ids,
        )

        image_sizes = []
        post_processed_targets = []
        post_processed_predictions = []

        # Process targets
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
            image_sizes.append(batch_image_sizes)

            for image_target in batch:
                boxes = torch.tensor(image_target["boxes"])
                boxes = convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
                labels = torch.tensor(image_target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})

        # Process predictions
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes)
            )
            post_processed_output = image_processor.post_process_object_detection(
                output, threshold=threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)

        # Compute metrics
        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

        # Safeguard against empty predictions or labels
        if len(post_processed_predictions) == 0 or len(post_processed_targets) == 0:
            return {"map": 0.0, "mar_100": 0.0}

        metric.update(post_processed_predictions, post_processed_targets)
        metrics = metric.compute()

        # Separate per-class metrics
        classes = metrics.pop("classes", [])
        map_per_class = metrics.pop("map_per_class", [])
        mar_100_per_class = metrics.pop("mar_100_per_class", [])

        # Ensure tensors are iterable
        def to_list(x):
            if isinstance(x, torch.Tensor):
                if x.dim() == 0:
                    return [x.item()]
                return x.tolist()
            return x

        classes = to_list(classes)
        map_per_class = to_list(map_per_class)
        mar_100_per_class = to_list(mar_100_per_class)

        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class
        ):
            class_name = id2label[class_id] if id2label is not None else class_id
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        # Round and convert all metrics to float
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}

        return metrics

    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=processor, id2label=id2label, threshold=0.01
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    training_args = TrainingArguments(
        output_dir="tt_finetuned",
        num_train_epochs=3,
        fp16=False,
        per_device_train_batch_size=8,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        # save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=False,
        report_to="none",
        logging_dir="./logs",
        logging_steps=10,  # Print every 10 steps
        logging_strategy="steps",  # Optional but explicit
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()

    model.save_pretrained("test")  # writes config.json + pytorch_model.bin
    processor.save_pretrained("test")  # writes preprocessor_config.json


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    train()
