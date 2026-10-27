"""Copy exllamav3's eval/model_diff.py so it can score our own token rows: with KL_ROWS=<file.safetensors> set,
get_test_tokens returns that file's `input_ids` instead of wikitext-2. Fails loudly if the upstream function moved.

    python model_diff_rows.patch.py SRC DST
"""
import sys

src, dst = sys.argv[1:]
text = open(src).read()
anchor = 'def get_test_tokens(tokenizer, rows, eval_len = 2048, eval_stride = 2048):\n'
assert text.count(anchor) == 1, "get_test_tokens signature changed upstream"
hook = (
    '    if os.environ.get("KL_ROWS"):\n'
    '        from safetensors.torch import load_file\n'
    '        ids = load_file(os.environ["KL_ROWS"])["input_ids"]\n'
    '        assert ids.shape[0] >= rows and ids.shape[1] >= eval_len, ids.shape\n'
    '        return ids[:rows, :eval_len].contiguous()\n'
)
open(dst, "w").write(text.replace(anchor, anchor + hook))
