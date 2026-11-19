# eval-only runner for extra checks (v2)
export PATH=$HOME/.local/bin:$PATH DEV=CPU:X86
echo "== extra v2"; python3 osmogrep_extra_check.py; a=$?
echo "== mypy"; python3 -m mypy --cache-dir /tmp/mypycache tinygrad/ > /tmp/mypy.txt 2>&1; b=$?; tail -5 /tmp/mypy.txt
echo "== ruff (PR files)"; python3 -m ruff check --no-cache tinygrad/engine/jit.py test/runtime/test_jit.py > /tmp/ruff.txt 2>&1; c=$?; tail -20 /tmp/ruff.txt
echo "extra=$a mypy=$b ruff=$c"; exit $((a|b|c))
