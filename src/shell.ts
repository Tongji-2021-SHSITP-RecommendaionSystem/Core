import "basic-type-extensions";
import { PythonShell } from "python-shell";
import { News } from "news-recommendation-entity";
import settings from "../settings.json";

interface SimpleNews {
	title: string;
	content: string;
}
export default class Recommender {
	public static script = "src/train.py";
	public batchSize: number;
	protected shell: PythonShell;
	protected available: boolean = false;
	protected curResult: number[] = new Array<number>();
	public constructor(batchSize: number) {
		this.batchSize = batchSize;
		this.launch();
	}
	public async execute(viewed: News[], candidates: News[]): Promise<number[]> {
		return new Promise(async (resolve, reject) => {
			if (Math.ceil(candidates.length / settings.model.candidatesPerBatch) != this.batchSize)
				reject(new Error(`Number of candidates should be within (${(this.batchSize - 1) * settings.model.candidatesPerBatch}, ${this.batchSize * settings.model.candidatesPerBatch}]`));
			let body: string = JSON.stringify({
				viewed: viewed.map(news => ({ title: news.title, content: news.content } as SimpleNews)),
				candidates: candidates.map(news => ({ title: news.title, content: news.content } as SimpleNews))
			});
			this.shell.send(`run ${body}`);
			Promise.wait(
				() => this.available,
				settings.model.timerInterval
			).then(() => {
				this.available = false;
				resolve(this.curResult)
			});
		})
	}
	public launch() {
		if (!this.shell || this.shell.terminated) {
			this.shell = new PythonShell(Recommender.script, {
				mode: "text",
				args: [this.batchSize.toString()],
				pythonPath: settings.model.pythonPath
			});
			this.shell.on("message", message => {
				this.curResult = JSON.parse(message);
				this.available = true;
			});
		}
	}
	public exit() {
		if (this.shell?.terminated == false)
			this.shell.send("exit").kill();
	}
}