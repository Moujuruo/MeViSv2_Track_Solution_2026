# Stage 1: Event Decomposition

Stage 1 converts each video-level target event into one or more instance-level grounding targets.
It uses an MLLM to:
- identify valid target instances;
- select the frame where each target is clearest;
- generate a discriminative description for each target.

## Files

- `run_ark_video_qa.py`  
  Run the Stage 1 query on a single video and target event.
- `batch_ark_video_qa.py`  
  Run Stage 1 in batch mode over a metadata file.
