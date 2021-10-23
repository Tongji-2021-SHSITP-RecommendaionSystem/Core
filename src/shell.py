from io import TextIOWrapper
import sys
import json
from jiagu import keywords as extract_keywords, summarize, sentiment as analyze_sentiment

from recommender import Recommender

recommender = Recommender(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

sys.stdout = TextIOWrapper(sys.stdout.buffer, encoding='utf8')
sys.stdin = TextIOWrapper(sys.stdin.buffer, encoding='utf8')
line: str
for line in sys.stdin:
	cmd, args = line.strip().split(' ', 1)
	args:list = json.loads(args)
	if cmd == 'exit':
		break
	elif cmd == 'recommend':
		print(recommender.calc_confidence(json.loads(args[0])))
	elif cmd == 'keywords':
		print(extract_keywords(args[0], int(args[1]) if args[1] != None else 5))
	elif cmd == "summary":
		print(summarize(args[0], int(args[1]) if args[1] != None else 5))
	elif cmd == "sentiment":
		sign, degree = analyze_sentiment(args[0])
		print(degree if sign == 'positive' else -degree)
	sys.stdout.flush()