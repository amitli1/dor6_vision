# Bounding box (50 images - validation set)
| Model             | #GT | #Pred | % Correct | Time (s)                  |
| ----------------  | --- | ----- |-----------|---------------------------|
| Molmo2-4B         | 180 | 114   | 63%       | 9:04 (Total on my laptop) |
| Gemma4-31B (1120) | 180 | 64    | 0.36%     | 20:00 (Total on server)   |
| ----------------  | --- | ----- |-----------| --------------------------|


# With molmo Crop (more GTs) (Validation):
|                | Anti aircraft | Launchers | Tank |
|----------------|---------------|-----------|------|
| Anti aircraft  | 9.09          | 8.08      | 6.06 |
| Launchers      | 0             | 33.33     | 3.92 |
| Tank           | 0             | 3.33      | 80   |


# With Gemma4 Crop (Validation):
|                | Anti aircraft | Launchers | Tank  |
|----------------|---------------|-----------|-------|
| Anti aircraft  | 2.02          | 5.058     | 3.03  |
| Launchers      | 0             | 29.41     | 0     |
| Tank           | 0             | 3.33      | 23.33 |





