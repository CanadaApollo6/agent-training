# How to work in this harness

You have exactly one tool: `ipython`. Every action is Python code in that REPL. Follow these rules.

## 1. Run shell commands with `bash()` inside `ipython`

There is no `bash` tool, no `edit` tool and no `websearch` tool. Calling them returns "Tool not found" and
wastes a turn. Put them inside an `ipython` call:

```python
r = await bash("ls -la /app && cat /app/README.md")
print(r.exit_code, r.output[-3000:])
```

Do not use `subprocess` or `os.system`. For a long command (a build, a test suite, a training run), start it and
keep the handle:

```python
build = bash("make -j8 2>&1")      # returns at once
# ... do other work, then later:
print(build.running, build.tail(40))
```

## 2. Change files with `edit`

For a change to an existing file, replace an exact string. Do not rewrite the whole file or use `sed -i`:

```python
old = '''    return a - b'''
new = '''    return a + b'''
await edit(path="/app/calc.py", old_str=old, new_str=new)
```

If `old_str` matches more than once, include more surrounding lines. Write new files with
`Path(...).write_text(...)`.

## 3. Watch your context and compact it

Your context window is limited. When it is more than half full and work remains, compact it:

```python
s = await compact.status()
print(s)                            # tokens, context_window, percent
if s["percent"] and s["percent"] > 50:
    await compact.run("keep: the task goal, the files changed, the failing test names, what to try next")
```

Compaction runs when your turn ends, and you resume automatically. Your Python variables survive it, so save
important results in variables or files before you compact. Check `compact.status()` every 15 turns or so,
and after any command that printed a lot of output.

## 4. Act every turn

Each reply has a length limit. If you think too long, the reply is cut off and the run ends with nothing done.
Keep your reasoning short and end each reply with an `ipython` call. If you are unsure, run a small experiment
instead of reasoning about it.

## 5. Use sub-agents for big independent pieces

For a separate, self-contained piece of work, such as researching an unfamiliar file format while you build
something else, start a child agent and keep working:

```python
child = await rlm.spawn("Find how the .xyz format stores timestamps. Reply with a short summary and code.",
                        name="format-research")
```

The child replies with a message that arrives on a later turn. Do not wait for it with `sleep`; end your turn
or continue other work. Do not spawn children for small steps you can do in one or two calls.

## 6. Check before you finish

Before you say you are done, run the task's own checks: the tests, the expected output, the requested files
and their exact paths. Fix what fails. Then state your final answer.
