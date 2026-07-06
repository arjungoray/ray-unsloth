# Hello SFT

## What
Train a tiny supervised adapter on three prompt/completion pairs using the new `ray_unsloth.recipes.sft` helpers.

## Why
This is the smallest supported example of the library layer: build datums, run epochs, save a checkpoint, and sample back from it.

## Expected Output
`run.py --smoke` prints one decreasing loss, writes a checkpoint, and samples a short completion from the saved weights.

## Cost
$0
