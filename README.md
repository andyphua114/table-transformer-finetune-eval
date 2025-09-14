# Table Transformer — Fine-tune & Evaluate

Utilities to fine-tune, run inference, and evaluate Microsoft’s Table Transformer (TATR) models on your own datasets (COCO format). The repository includes simple training loops for table detection and table structure recognition, plus COCO-style mAP evaluation and a minimal inference script.

```text
table-transformer-finetune-eval/
├── finetune.py               # Fine-tune script for table transformer detection
├── finetune_structure.py     # Fine-tune for table transformer structure recognition
├── model_inference.py        # Run inference with a fine-tuned model
├── metric_evaluation.py      # Evaluate predictions using COCO mAP
├── custom.json               # Example of ground truth for COCO mAP evaluation
├── finetuned_pred.json       # Example of fine-tuned model predictions for COCO mAP evaluation
└── README.md
```

## Key Requirements

- 3.11
- PyTorch + TorchVision
- Hugging Face transformers, datasets
- pycocotools

## Blog Article

You can find a guide for implementation at the following link: [How to Fine-Tune Table Transformer on Your Own Domain-Specific Data](https://medium.com/@andyphuawc/how-to-fine-tune-table-transformer-on-your-own-domain-specific-data-e8a04a7d41f0#594c)

## Notes

Written based on a Windows machine, so do take note when working with directory path.