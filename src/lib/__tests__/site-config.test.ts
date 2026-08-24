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
    process.env.NEXT_PUBLIC_BASE_PATH = '/llm-explorer';

    const { withBasePath } = await import('@/lib/site-config');

    expect(withBasePath('/favicon.ico')).toBe('/llm-explorer/favicon.ico');
    expect(withBasePath('/')).toBe('/llm-explorer');
  });

  it('uses the public LABIIUM domain at the site root by default', async () => {
    delete process.env.NEXT_PUBLIC_BASE_PATH;
    delete process.env.NEXT_PUBLIC_SITE_URL;

    const { absoluteUrl, withBasePath } = await import('@/lib/site-config');

    expect(withBasePath('/')).toBe('/');
    expect(absoluteUrl('/')).toBe('https://llm.labiium.com/');
  });

  it('builds absolute URLs from the configured site origin', async () => {
    process.env.NEXT_PUBLIC_BASE_PATH = '/llm-explorer';
    process.env.NEXT_PUBLIC_SITE_URL = 'https://example.com/';

    const { absoluteUrl, resolveAssetUrl } = await import('@/lib/site-config');

    expect(absoluteUrl('/')).toBe('https://example.com/llm-explorer');
    expect(resolveAssetUrl('/images/large-og.png')).toBe(
      'https://example.com/llm-explorer/images/large-og.png',
    );
    expect(resolveAssetUrl('https://cdn.example.com/og.png')).toBe(
      'https://cdn.example.com/og.png',
    );
  });
});
