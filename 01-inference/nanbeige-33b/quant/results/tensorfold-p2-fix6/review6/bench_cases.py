import asyncio
import json
from pathlib import Path
import independent as m
from extended import case
m.ROOT=Path(__file__).parent/'bench'
m.ROOT.mkdir(exist_ok=True)
def cases():
    for repeat in range(5):
        for count in [1,8,32]:
            for style,value in [('plain',"'☃'"),('named',"'\\N{SNOWMAN}'")]:
                yield case(f'{repeat}-{count}-{style}','bench',[(f'p{i}',value) for i in range(count)])
m.new_cases=cases
asyncio.run(m.generate())
