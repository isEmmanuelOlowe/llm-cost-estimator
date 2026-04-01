describe('site config helpers', () => {
  const originalBasePath = process.env.NEXT_PUBLIC_BASE_PATH;
  const originalSiteUrl = process.env.NEXT_PUBLIC_SITE_URL;

  afterEach(() => {
    if (typeof originalBasePath === 'undefined') {
      delete process.env.NEXT_PUBLIC_BASE_PATH;
    } else {
      process.env.NEXT_PUBLIC_BASE_PATH = originalBasePath;
    }

    if (typeof originalSiteUrl === 'undefined') {
      delete process.env.NEXT_PUBLIC_SITE_URL;
    } else {
      process.env.NEXT_PUBLIC_SITE_URL = originalSiteUrl;
    }

    jest.resetModules();
  });

  it('prefixes paths with the configured base path', async () => {
    process.env.NEXT_PUBLIC_BASE_PATH = '/llm-cost-estimator';

    const { withBasePath } = await import('@/lib/site-config');

    expect(withBasePath('/favicon/favicon.ico')).toBe(
      '/llm-cost-estimator/favicon/favicon.ico',
    );
    expect(withBasePath('/')).toBe('/llm-cost-estimator');
  });

  it('builds absolute URLs from the configured site origin', async () => {
    process.env.NEXT_PUBLIC_BASE_PATH = '/llm-cost-estimator';
    process.env.NEXT_PUBLIC_SITE_URL = 'https://example.com/';

    const { absoluteUrl, resolveAssetUrl } = await import('@/lib/site-config');

    expect(absoluteUrl('/')).toBe('https://example.com/llm-cost-estimator');
    expect(resolveAssetUrl('/images/large-og.png')).toBe(
      'https://example.com/llm-cost-estimator/images/large-og.png',
    );
    expect(resolveAssetUrl('https://cdn.example.com/og.png')).toBe(
      'https://cdn.example.com/og.png',
    );
  });
});
