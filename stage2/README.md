# Stage 2: SAM3-agent Grounding and Propagation

Stage 2 takes the outputs of Stage 1 and produces seed masks on selected frames, then propagates
those masks through the full video.

## Files

- `run_sam3_agent_from_batch_result.py`  
  Run SAM3-agent on one Stage 1 JSON result.
- `batch_run_sam3_agent_from_outputs.py`  
  Batch version of SAM3-agent grounding.
- `generate_sam3_masks_from_agent.py`  
  Propagate masks from SAM3-agent outputs through videos.
- `generate_sam3_masks_from_bbox.py`  
  Utilities for bbox-based SAM3 segmentation and tracker propagation. This file is kept because
  later scripts reuse its tracker and propagation helpers.


