import fs from 'node:fs';
import path from 'node:path';

const outDir = path.join(process.cwd(), 'out');
const requiredFiles = ['index.html', '404.html', 'robots.txt', 'sitemap.xml'];
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';
const siteOrigin = (
  process.env.NEXT_PUBLIC_SITE_URL || 'https://isemmanuelolowe.github.io'
).replace(/\/+$/, '');
const siteUrl = `${siteOrigin}${basePath}`;

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function readFile(relativePath) {
  return fs.readFileSync(path.join(outDir, relativePath), 'utf8');
}

for (const file of requiredFiles) {
  assert(
    fs.existsSync(path.join(outDir, file)),
    `Missing export artifact: ${file}`,
  );
}

const indexHtml = readFile('index.html');
const sitemapXml = readFile('sitemap.xml');
const robotsTxt = readFile('robots.txt');

assert(
  !indexHtml.includes('https:/prohpet.ai'),
  'Found malformed legacy site URL',
);
assert(
  !sitemapXml.includes('https:/prohpet.ai'),
  'Found malformed legacy sitemap URL',
);
assert(
  indexHtml.includes(siteUrl),
  `Expected exported HTML to reference ${siteUrl}`,
);
assert(
  sitemapXml.includes(siteUrl),
  `Expected sitemap to reference ${siteUrl}`,
);
assert(
  robotsTxt.includes('Sitemap:'),
  'robots.txt is missing a sitemap reference',
);

if (basePath) {
  const requiredBasePathFragments = [
    `${basePath}/favicon/favicon.ico`,
    `${basePath}/favicon/site.webmanifest`,
    `${basePath}/fonts/inter-var-latin.woff2`,
  ];

  for (const fragment of requiredBasePathFragments) {
    assert(
      indexHtml.includes(fragment),
      `Expected exported HTML to include basePath-safe asset reference: ${fragment}`,
    );
  }

  const forbiddenRootRelativeFragments = [
    'href="/favicon/favicon.ico"',
    'href="/favicon/site.webmanifest"',
    'href="/fonts/inter-var-latin.woff2"',
  ];

  for (const fragment of forbiddenRootRelativeFragments) {
    assert(
      !indexHtml.includes(fragment),
      `Found root-relative asset reference that breaks GitHub Pages: ${fragment}`,
    );
  }
}

console.log('Static export verification passed.');
