# Object detection:
1. Molmo2-4B (point) detect more objects than Gemma4-31B (1120 tokens)
2. Example: VBS_Record_3/frame_294_00_06_912.jpg

# Gemma4-31B
1. with 1120 tokens we find more objcets (more frames with objects and each frame it detects more objects) than 560/280 tokens

# Use case: Video 3, File: frame_273_00_06_418.jpg
1. Molmo found 2 objects, Gemma found 6 objects
2. Molmo run ~2sec and Gemma run ~10 seconds
3. Gemma with 280 tokens - didnt find any objects
4. Gemma with 560 tokens - found 4 objects (Took: ~6.5Seconds)

# Videos (GT)
1. Video 1:
- sa-6
- trucks
- humers
- t-90 (end)
- grad (tunnel)

  2. Video 2:
  - sa-22