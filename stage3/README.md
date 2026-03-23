# Stage 3: Refinement and Final Export

Stage 3 revisits problematic predictions, performs behavior-level verification when needed,
selects the best available results, and builds the final submission structure.

## Files

- `scan_agent_badcases.py`  
  Inspect predictions and flag empty masks or highly overlapping masks.
- `review_keyword_prompt_behavior.py`  
  Sample frames, highlight mask boundaries, and ask an MLLM to check whether the tracked target
  matches the original event.
- `rerun_refined_qa_for_badcases.py`  
  Regenerate refined descriptions for flagged cases.
- `propagate_best_agent_outputs.py`  
  Choose the best available stage output and propagate the corresponding masks.
- `build_submission_text_from_final_masks.py`  
  Export the final merged masks into the required submission structure.


