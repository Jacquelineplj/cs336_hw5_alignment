uv run python3 cs336_alignment/grpo.py 3e-5 --use-masked-mean && \
uv run python3 cs336_alignment/grpo.py 3e-5 --no-use-masked-mean && \
uv run python3 cs336_alignment/grpo.py 2e-5 --use-masked-mean --epochs-per-rollout-batch 2 --train-batch-size 128 && \
