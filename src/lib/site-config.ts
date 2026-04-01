const DEFAULT_SITE_NAME = 'LLM Cost Estimator';
const DEFAULT_SITE_DESCRIPTION =
  'Estimate LLM memory fit, serving topology, and hosting costs for modern open models.';
const DEFAULT_SITE_ORIGIN = 'https://isemmanuelolowe.github.io';
const DEFAULT_BASE_PATH = '';
const DEFAULT_OG_IMAGE_PATH = '/images/large-og.png';

function isAbsoluteUrl(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function normalizeOrigin(origin: string): string {
  return origin.replace(/\/+$/, '');
}

function normalizePath(path: string): string {
  if (!path || path === '/') return '/';
  return path.startsWith('/') ? path : `/${path}`;
}

export const siteName = DEFAULT_SITE_NAME;
export const siteDescription = DEFAULT_SITE_DESCRIPTION;
export const siteOrigin = normalizeOrigin(
  process.env.NEXT_PUBLIC_SITE_URL || DEFAULT_SITE_ORIGIN,
);
export const siteBasePath =
  process.env.NEXT_PUBLIC_BASE_PATH || DEFAULT_BASE_PATH;

export function withBasePath(path = '/'): string {
  const normalizedPath = normalizePath(path);

  if (!siteBasePath) return normalizedPath;
  if (normalizedPath === '/') return siteBasePath;

  return `${siteBasePath}${normalizedPath}`;
}

export function absoluteUrl(path = '/'): string {
  return `${siteOrigin}${withBasePath(path)}`;
}

export function resolveAssetUrl(path = DEFAULT_OG_IMAGE_PATH): string {
  if (isAbsoluteUrl(path)) return path;
  return absoluteUrl(path);
}
