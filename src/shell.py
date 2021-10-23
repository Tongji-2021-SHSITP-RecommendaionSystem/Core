from io import TextIOWrapper
import sys
import json

from recommender import Recommender

recommender = Recommender(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding='utf8')
sys.stdin = TextIOWrapper(sys.stdin.buffer, encoding='utf8')
line: str
for line in sys.stdin:
	args = line.strip().split(' ', 1)
	if args[0] == 'exit':
		break
	elif args[0] == 'run':
		print(recommender.calc_confidence(json.loads(args[1])))
		sys.stdout.flush()