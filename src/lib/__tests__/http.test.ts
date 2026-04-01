import { fetchJson, HttpError } from '@/lib/http';

describe('fetchJson', () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    jest.restoreAllMocks();
    if (originalFetch) {
      globalThis.fetch = originalFetch;
    } else {
      Reflect.deleteProperty(globalThis, 'fetch');
    }
  });

  it('returns parsed JSON for successful responses', async () => {
    const fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ model: 'demo' }),
    });
    globalThis.fetch = fetchMock as typeof fetch;

    await expect(fetchJson<{ model: string }>('/api/demo')).resolves.toEqual({
      model: 'demo',
    });
  });

  it('throws HttpError for non-OK responses', async () => {
    const fetchMock = jest.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ message: 'missing' }),
    });
    globalThis.fetch = fetchMock as typeof fetch;

    await expect(fetchJson('/api/missing')).rejects.toEqual(
      new HttpError('Request failed with status 404', 404, '/api/missing'),
    );
  });

  it('passes an abort signal to fetch when a timeout is configured', async () => {
    const fetchMock = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
    globalThis.fetch = fetchMock as typeof fetch;

    await fetchJson('/api/slow', { timeoutMs: 5 });

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/slow',
      expect.objectContaining({
        signal: expect.any(AbortSignal),
      }),
    );
  });
});
