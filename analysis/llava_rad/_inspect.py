import json
from collections import Counter
r=json.load(open(r'~/AppData/Local/Temp/claude/E--Coding-Git-TFM--claude-worktrees-strange-yalow-4533b2/45b30dfb-0020-43ff-886f-c1299ee05ed5/scratchpad/per_finding.json'))
print('records:',len(r))
print('keys:',list(r[0].keys()))
print('labels:',dict(Counter(x['label'] for x in r)))
print('findings:',Counter(x['finding'] for x in r).most_common(10))
print('hm_len:',len(r[0]['hm']))
print('unique_ids:',len(set(x['id'] for x in r)))
