"""Verify all published bytes before trusting model files. Standard library only."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parent
def verify():
    manifest=json.loads((ROOT/'ARTIFACTS.json').read_text(encoding='utf-8'));bad=[]
    for item in manifest['files']:
        path=(ROOT/item['file']).resolve()
        if not path.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe artifact path.')
        actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual!=item['sha256']:bad.append(item['file'])
    result={'status':'PASS' if not bad else 'FAIL','checked_files':len(manifest['files']),'failures':bad,'scope':'Byte integrity; no fitting, accuracy evaluation, NASA download or public-host verification.'}
    print(json.dumps(result));return not bad
if __name__=='__main__':sys.exit(0 if verify() else 1)
