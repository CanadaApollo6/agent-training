import ast
import asyncio
import collections
import hashlib
import itertools
import json
from pathlib import Path
import random
import warnings
import unicodedata as ud
import independent as m

warnings.filterwarnings('ignore', category=SyntaxWarning)
BASE = Path(__file__).parent
m.ROOT = BASE / 'extended'
m.ROOT.mkdir(exist_ok=True)
COUNTS = collections.Counter()
META = []

def literal(name):
    return "'\\N{" + name + "}'"

def accepted(value):
    try:
        ast.literal_eval(value)
        return True
    except (SyntaxError, ValueError):
        return False

def variants(name):
    return (name, name.lower(), ''.join(c.lower() if i % 2 else c for i,c in enumerate(name)))

def source(name):
    raw = (BASE / 'ucd' / name).read_bytes()
    return [l.split(';') for l in raw.decode().splitlines() if l and not l.startswith('#')]

def case(label, group, params):
    props = {p: {'type': 'object'} for p,v in params}
    return {'name': label, 'group': group, 'raw': '</think>' + m.xml('probe', params),
        'request': {'messages': [{'role':'user','content':'Check Unicode literals.'}],
            'tools':[{'type':'function','function':{'name':'probe','parameters':{'type':'object','properties':props}}}],
            'max_tokens':65000, 'stream_options':{'include_usage': True}}}

def cases():
    assert ud.unidata_version == '15.0.0'
    good, bad = [], []
    with (m.ROOT / 'names.tsv').open('w') as tsv:
        for cp in range(0x110000):
            COUNTS['codepoints_visited'] += 1
            name = ud.name(chr(cp), None)
            if name is None: continue
            COUNTS['named_codepoints'] += 1
            for variant in variants(name):
                s = literal(variant)
                yes = accepted(s)
                COUNTS['named_variants'] += 1
                (good if yes else bad).append(s)
                tsv.write(f'{cp if yes else -1}\t{variant}\n')
        for cp,name,kind in source('NameAliases.txt'):
            COUNTS['aliases'] += 1
            for variant in variants(name):
                s = literal(variant)
                yes = accepted(s)
                COUNTS['alias_variants'] += 1
                (good if yes else bad).append(s)
                tsv.write(f'{int(cp,16) if yes else -1}\t{variant}\n')
        for name,points in source('NamedSequences.txt'):
            COUNTS['named_sequences'] += 1
            for variant in variants(name):
                s = literal(variant)
                assert not accepted(s)
                bad.append(s)
                tsv.write(f'-1\t{variant}\n')
    COUNTS['valid_name_literals'] = len(good)
    COUNTS['invalid_name_literals'] = len(bad)
    for i in range(0,len(good),128):
        yield case(f'valid-names-{i//128:05}', 'valid-names', [('payload','['+','.join(good[i:i+128])+']')])
    for i in range(0,len(bad),12):
        yield case(f'invalid-names-{i//12:05}', 'invalid-names', [(f'p{j}', s) for j,s in enumerate(bad[i:i+12])])
    names = ['', ' ', 'SNOWMAN ', ' SNOWMAN', 'SNOW MAN', 'SNOW-MAN', 'SNOW_MAN', 'SNOWMAN\t', 'SNOWMAN\n',
        'SNOWMAN\r', 'SNOWMAN\x00', 'SNOWMAN\u00a0', 'ＳＮＯＷＭＡＮ', 'SNOWMAN'+' '*50000,
        'SNOWMAN}x{', 'LATIN CAPITAL LETTER A WITH RING ABOVE', 'LATIN  CAPITAL LETTER A WITH RING ABOVE',
        'HANGUL SYLLABLE GA', 'HANGUL SYLLABLE GAG', 'HANGUL SYLLABLE HIH', 'HANGUL SYLLABLE A',
        'HANGUL SYLLABLE', 'HANGUL SYLLABLE ', 'HANGUL SYLLABLE GAX', 'HANGUL SYLLABLE G',
        'HANGUL SYLLABLE GGAG', 'HANGUL SYLLABLE GAE', 'HANGUL SYLLABLE GGAE', 'HANGUL SYLLABLE A'+ 'G'*50000,
        'TANGUT IDEOGRAPH-17000', 'LATIN CAPITAL LETTER A\u2010', 'CJK UNIFIED IDEOGRAPH\u20104E00',
        'CJK UNIFIED IDEOGRAPH\u22124E00', 'CJK UNIFIED IDEOGRAPH--4E00', 'NULL', 'NUL', 'BOM', 'ZWNBSP',
        'LATIN SMALL LETTER GHA', 'KAKTOVIK NUMERAL ZERO', 'TROLL', 'PINK HEART']
    bounds = [0x33ff,0x3400,0x4dbf,0x4dc0,0x4dff,0x4e00,0x9fff,0xa000,0x1ffff,0x20000,0x2a6df,0x2a6e0,
        0x2a6ff,0x2a700,0x2b739,0x2b73a,0x2b73f,0x2b740,0x2b81d,0x2b81e,0x2b81f,0x2b820,
        0x2cea1,0x2cea2,0x2ceaf,0x2ceb0,0x2ebe0,0x2ebe1,0x2ebf0,0x2ee5d,0x2ffff,0x30000,
        0x3134a,0x3134b,0x3134f,0x31350,0x323af,0x323b0,0x33479,0x10ffff,0x110000]
    for cp in bounds:
        for suffix in [f'{cp:X}',f'{cp:x}',f'{cp:04X}',f'{cp:05X}',f'{cp:06X}',f'{cp:08X}',
                       '+'+f'{cp:X}','-'+f'{cp:X}',f'{cp:X}'+'_','0x'+f'{cp:X}']:
            names.append('CJK UNIFIED IDEOGRAPH-'+suffix)
    for p, cps in [('NUSHU CHARACTER-', [0x1b16f,0x1b170,0x1b2fb,0x1b2fc]),
                   ('KHITAN SMALL SCRIPT CHARACTER-', [0x18aff,0x18b00,0x18cd5,0x18cd6])]:
        for cp in cps:
            for n in [f'{cp:X}',f'{cp:x}',f'{cp:06X}',f'{cp:04X}','+'+f'{cp:X}']:
                names.extend([p+n,(p+n).lower()])
    newer = BASE/'newer-names.json'
    if newer.exists(): names.extend(json.loads(newer.read_text())['names'])
    values = [literal(n) for n in names]
    values += ["'\\N'", "'\\N{'", "'\\N{}'", "'\\N{SNOWMAN'", "'\\N{SNOWMAN}'", "'\\N{SNOWMAN}x'", "'\\N{SNOWMAN\\}'", "'\\N{SNOWMAN}\\N{NULL}'"]
    for esc, n in [('x',2),('u',4),('U',8)]:
        for size in range(n+2):
            for digits in ['0'*size,'f'*size,'+'+'0'*size,'_'+'0'*size,'0'*size+'_', '-'+'0'*size]:
                for tail in ["'",'',"' 'end'",'\\\n\'']:
                    values.append("'\\"+esc+digits+tail)
    for esc in ['x00','x7f','x80','xff','ud800','udfff','uD800','u0000','U0010ffff','U00110000','Uffffffff','U80000000','U0000d800','ud83d\\ude00']:
        values += ["'\\"+esc+"'",'"\\'+esc+'"', "{'x':'\\"+esc+"'}"]
    for val in [0,1,7,8,63,127,255,256,511,512,777]:
        values += ["'\\"+format(val,'o')+"'", "'\\"+format(val,'o')+"0'", "r'\\"+format(val,'o')+"'"]
    for prefix in ['','r','R','u','U','b','B','br','rb','Br','rB','f','F','fr','rf','ur','ru','rr','uu']:
        for q in ["'", '"', "'''", '"""']:
            for s in ['abc','\\N{SNOWMAN}', '\\u0041','\\x+1','\\777', '\\\n', '\\\r\n', '\\'+q, 'a\x00b', '{1+2}', '雪']:
                values.append(prefix+q+s+q)
    values += ["'a' 'b'", "r'\\n' u'\\N{SNOWMAN}'", "('a'\n'b')", "'a'\n'b'", "['a' # x\n'b']", "'a'\\\n'b'",
        "'''a\\\nb'''", "'''a\\N{SNOWMAN}\n'''", '"""a\\\r\nb"""', "[1,2]\x00", "\x00[1,2]", "[1,#\x00\n2]",
        "{'x':b'abc'}", "{'x':set()}", "{'x':{1,2}}", "{'x':...}", "{'x':1+2j}", "{b'abc': 1}", "{(1,2): 1}",
        "b'' == b''", "{'x': (1,True,None)}", "{1:'a',True:'b',1.0:'c'}", "{'x':0x_10, 'y':+0b_1}",
        "{'x': 'a' u'b' r'\\n'}", '[True,None,False,]', "[1,# hi\n2]", "(- (1))", "[1\x0c,2]", '\ufeff[1]',
        "{'x':'\\N{SNOWMAN}', 'y':b'bytes'}", "{'x': b'bytes', 'y':'\\N{SNOWMAN}'}"]
    for i,v in enumerate(dict.fromkeys(values)):
        COUNTS['edge_literals'] += 1
        META.append({'name':f'edge-{i:05}','literal':v})
        yield case(f'edge-{i:05}', 'edge', [('payload',v)])
    (m.ROOT/'case-meta.json').write_text(json.dumps(META, ensure_ascii=True))
    (m.ROOT/'coverage.json').write_text(json.dumps(dict(COUNTS),indent=2))

if __name__ == '__main__':
    m.new_cases = cases
    asyncio.run(m.generate())
