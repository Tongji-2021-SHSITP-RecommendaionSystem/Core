import "basic-type-extensions";
import { PythonShell } from "python-shell";
import { News } from "news-recommendation-entity";
import settings from "../settings.json";

interface SimpleNews {
	title: string;
	content: string;
}
export default class Shell {
	public static script = `${__dirname}/shell.py`;
	public batchSize?: number;
	protected shell: PythonShell;
	protected available: boolean = false;
	protected result: string;
	private _busy: boolean = false;

	public get busy() { return this._busy; }

	public constructor(batchSize?: number) {
		this.batchSize = batchSize;
		this.launch();
	}

	public launch() {
		if (!this.shell || this.shell.terminated) {
			this.shell = new PythonShell(Shell.script, {
				mode: "text",
				args: this.batchSize ? [this.batchSize.toString()] : [],
				pythonPath: `${__dirname}/../.venv/Scripts/python.exe`
			});
			this.shell.on("message", message => {
				this.result = message;
				this.available = true;
			});
			this.shell.on("pythonError", console.log);
			this.shell.on("stderr", console.log);
			this.shell.on("error", console.log);
		}
	}
	public exit() {
		if (this.shell?.terminated == false)
			this.shell.send("exit").kill();
	}
	public async recommend(viewed: News[], candidates: News[]): Promise<number[]> {
		return new Promise(async (resolve, reject) => {
			if (Math.ceil(candidates.length / settings.model.candidatesPerBatch) != this.batchSize)
				reject(new Error(`Number of candidates should be within (${(this.batchSize - 1) * settings.model.candidatesPerBatch}, ${this.batchSize * settings.model.candidatesPerBatch}]`));
			let body: string = JSON.stringify({
				viewed: viewed.map(news => ({ title: news.title, content: news.content } as SimpleNews)),
				candidates: candidates.map(news => ({ title: news.title, content: news.content } as SimpleNews))
			});
			resolve(JSON.parse(await this.execute("recommend", [body])));
		})
	}
	public keywords(content: string, count: number = 5): Promise<string[]> {
		return this.execute("keywords", [content, count]).then(JSON.parse);
	}
	public summary(content: string, count: number = 5): Promise<string[]> {
		return this.execute("summary", [content, count]).then(JSON.parse);
	}
	public sentiment(content: string): Promise<number> {
		return this.execute("sentiment", [content]).then(Number);
	}
	private async execute(cmd: string, args: any[]): Promise<string> {
		this._busy = true;
		this.shell.send(`${cmd} ${JSON.stringify(args)}`);
		return new Promise(resolve => {
			Promise.wait(
				() => this.available,
				settings.model.timerInterval
			).then(() => {
				this.available = false;
				this._busy = false;
				resolve(this.result)
			});
		});
	}
}