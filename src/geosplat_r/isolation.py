"""Read auditing for explicit train assets; not a security sandbox."""
import os
import sys
from pathlib import Path


def audit_training_reads(data):
    data=Path(data).resolve();accessed=[]
    def audit(event,arguments):
        if event=='open' and isinstance(arguments[0],(str,bytes)):
            p=Path(os.fsdecode(arguments[0])).resolve()
            if data.parent in p.parents and not (data==p or data in p.parents):
                raise RuntimeError(f'Non-training asset access: {p}')
            if data in p.parents:accessed.append(str(p.relative_to(data)))
    sys.addaudithook(audit)
    return accessed
