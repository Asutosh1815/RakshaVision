# ⚖️ Deep Learning Checkpoints & Model Weights

This directory houses the neural network weights utilized by RakshaVision:

- `yolov8n.pt`: Base person detection model for worker localization.
- `yolov8n-mendeley-ppe.pt`: Fine-tuned YOLOv8 model trained on 2,286 Mendeley industrial PPE images for high-accuracy vest and helmet detection.
- `yolov8n-ppe.pt`: Dedicated PPE detection checkpoint for protective footwear (shoes/boots) and industrial gloves.
- `yolov8n-hardhat.pt`: Specialized headwear classification checkpoint.
