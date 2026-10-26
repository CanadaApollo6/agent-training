"""Registers Nanbeige4.3's chat format in an installed SpecForge (appends to specforge/data/template.py; idempotent).
The model's own HF template renders the text (ThinkingParser); these strings only find the assistant spans to train on.
System prompt None: the HF template adds Nanbeige's default system line itself.

    python register_template.py
"""
import specforge.data.template as t

if "nanbeige-thinking" not in t.TEMPLATE_REGISTRY.templates:
    open(t.__file__, "a").write('''

TEMPLATE_REGISTRY.register(
    name="nanbeige-thinking",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\\n",
        user_header="<|im_start|>user\\n",
        system_prompt=None,
        end_of_turn_token="<|im_end|>\\n",
        parser_type="thinking",
        enable_thinking=True,
    ),
)
''')
print("nanbeige-thinking registered")
