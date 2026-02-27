#!/bin/bash
while true; do
    echo ""
    echo "=== $(date) ==="
    LATEST=$(ls -dt ./whisper-aviation-model-medium/checkpoint-* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "Checkpoint: $LATEST"
        cp -r "$LATEST" ./whisper-aviation-model-medium-final
        python evaluate_on_testset.py | grep -E "(WER|ratio|RESULTS|samples)"
    else
        echo "Waiting for checkpoint..."
    fi
    sleep 600
done
