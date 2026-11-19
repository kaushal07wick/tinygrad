# eval-only runner for extra checks
export PATH=$HOME/.local/bin:$PATH DEV=CPU:X86
echo "== extra"; python3 osmogrep_extra_check.py; a=$?
echo "== frozen reproducer"; python3 osmogrep_repro_jit_return_input.py
echo "== mypy"; python3 -m mypy tinygrad/ > /tmp/mypy.txt 2>&1; b=$?; tail -15 /tmp/mypy.txt
echo "== ruff"; python3 -m ruff check . > /tmp/ruff.txt 2>&1; c=$?; tail -15 /tmp/ruff.txt
echo "== skipped"; python3 -m unittest -v test.runtime.test_jit test.runtime.test_symbolic_jit test.runtime.test_pickle 2>&1 | grep -E "skipped|^Ran|^OK|^FAILED"
echo "extra=$a mypy=$b ruff=$c"; exit $((a|b|c))
