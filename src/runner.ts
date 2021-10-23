import "basic-type-extensions";
import { News } from "news-recommendation-entity";
import Shell from "./shell";
import settings from "../settings.json";

export default class Runner {
	protected shells: Map<number, Shell>;
	public constructor() {
		this.shells = new Map();
	}
	public async recommend(viewed: News[], candidates: News[]): Promise<Array<[News, number]>> {
		return new Promise(async (resolve, reject) => {
			viewed = viewed.shuffle().slice(0, settings.model.maxViewed);
			const batchSize = Math.ceil(candidates.length / settings.model.candidatesPerBatch);
			this.getShell(batchSize).then(shell => {
				shell.recommend(viewed, candidates).then(
					confidence => {
						const result = new Array<[News, number]>(candidates.length);
						for (let i = 0; i < candidates.length; ++i)
							result[i] = [candidates[i], confidence[i]];
						resolve(result.keySort(member => member[1]).reverse());
					}, reject
				)
			});
		})
	}
	public launch(): void {
		this.shells.forEach(shell => shell.launch());
	}
	public exit(): void {
		this.shells.forEach(shell => shell.exit());
	}
	public keywords(content: string, count: number = 5): Promise<string[]> {
		return this.getShell().then(shell => shell.keywords(content, count));
	}
	public summary(content: string, count: number = 5): Promise<string[]> {
		return this.getShell().then(shell => shell.summary(content, count));
	}
	public sentiment(content: string): Promise<number> {
		return this.getShell().then(shell => shell.sentiment(content));
	}
	protected getShell(): Promise<Shell>;
	protected getShell(batchSize: number): Promise<Shell>;
	protected async getShell(batchSize?: number): Promise<Shell> {
		if (batchSize == null) {
			await Promise.wait(() => {
				for (const shell of this.shells.values())
					if (!shell.busy)
						return true;
				return false;
			}, settings.model.timerInterval);
			for (const shell of this.shells.values())
				if (!shell.busy)
					return shell;
		}
		if (this.shells.has(batchSize)) {
			const shell = this.shells.get(batchSize);
			if (shell.busy)
				await Promise.wait(() => !shell.busy, settings.model.timerInterval);
			return shell;
		}
		else {
			const shell = new Shell(batchSize);
			this.shells.set(batchSize, shell);
			return shell;
		}
	}
}