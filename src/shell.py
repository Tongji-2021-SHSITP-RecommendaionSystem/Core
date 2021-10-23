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
	args = line.strip().split(' ', 1)
	if args[0] == 'exit':
		break
	elif args[0] == 'recommend':
		print(recommender.calc_confidence(json.loads(args[1])))
	elif args[0] == 'keywords':
		print(extract_keywords(args[1], int(args[2]) if args[2] != None else 5))
	elif args[0] == "summary":
		print(summarize(args[1], int(args[2]) if args[2] != None else 5))
	elif args[0] == "sentiment":
		sign, degree = analyze_sentiment(args[1])
		print(degree if sign == 'positive' else -degree)
	sys.stdout.flush()