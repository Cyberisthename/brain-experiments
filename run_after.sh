while pgrep -f "python3 round3.py" >/dev/null; do sleep 30; done
python3 round3_lr_sanity.py > round3_lr.log 2>&1
