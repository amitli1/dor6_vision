# Notes (model_family):
1. NOT_IN_GT - model found an object that is not in the GT 
2. Not_Found - model didn't find the object in the image

# Bounding box (50 images - validation set) (bb_threshold=0.4)
| Model             | #GT | #Pred | % Correct   | Time (s)                  |
| ----------------  | --- |-------|-------------|---------------------------|
| Molmo2-4B         | 180 | 115   | 64%         | 9:04 (Total on my laptop) |
| Gemma4-31B (1120) | 180 | 120   | 67%         | 20:00 (Total on server)   |
| ----------------  | --- | ----- | ----------- | --------------------------|


# With molmo Crop (more GTs) (Validation):
|                | Anti aircraft | Launchers | Tank  |
|----------------|---------------|-----------|-------|
| Anti aircraft  | 15.0          | 13.33     | 10.00 |
| Launchers      | 0.0           | 65.38     | 7.69  |
| Tank           | 0.0           | 3.57      | 85.71 |


# With Gemma4 Crop (Validation):
|                | Anti aircraft | Launchers | Tank  |
|----------------|---------------|-----------|-------|
| Anti aircraft  | 5.71          | 14.29     | 8.57  |
| Launchers      | 0             | 71.43     | 0     |
| Tank           | 0             | 0.00      | 87.50 |





