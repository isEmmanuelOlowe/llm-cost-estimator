export interface ModelSearchRecord {
  id: string;
  label: string;
  family?: string | null;
  summary?: string | null;
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function subsequenceScore(query: string, value: string): number {
  if (!query) return 0;
  let queryIndex = 0;
  let matched = 0;
  for (const character of value) {
    if (character === query[queryIndex]) {
      queryIndex += 1;
      matched += 1;
      if (queryIndex === query.length) break;
    }
  }
  return matched === query.length ? matched / value.length : 0;
}

function scoreModel(model: ModelSearchRecord, query: string): number {
  const normalizedQuery = normalize(query);
  const id = normalize(model.id);
  const label = normalize(model.label);
  const family = normalize(model.family ?? '');
  const summary = normalize(model.summary ?? '');
  const fields = [id, label, family, summary];
  const terms = normalizedQuery.split(/\s+/).filter(Boolean);
  if (!terms.length) return 0;

  let score = 0;
  for (const term of terms) {
    const fieldScore = Math.max(
      ...fields.map((field) => {
        if (field === term) return 1000;
        if (field.startsWith(term)) return 700;
        if (field.includes(term)) return 500;
        return subsequenceScore(term, field) * 100;
      }),
    );
    if (fieldScore <= 0) return 0;
    score += fieldScore;
  }
  if (id === normalizedQuery) score += 2000;
  else if (label === normalizedQuery) score += 1500;
  return score;
}

export function fuzzySearchModels<T extends ModelSearchRecord>(
  models: T[],
  query: string,
  limit = 8,
): T[] {
  if (!normalize(query)) return models.slice(0, limit);
  return models
    .map((model, index) => ({ model, index, score: scoreModel(model, query) }))
    .filter((entry) => entry.score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, limit)
    .map((entry) => entry.model);
}
