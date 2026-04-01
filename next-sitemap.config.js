/**
 * @type {import('next-sitemap').IConfig}
 * @see https://github.com/iamvishnusankar/next-sitemap#readme
 */
const siteOrigin = (
  process.env.NEXT_PUBLIC_SITE_URL || 'https://isemmanuelolowe.github.io'
).replace(/\/+$/, '');
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || '';

module.exports = {
  siteUrl: `${siteOrigin}${basePath}`,
  outDir: 'out',
  generateRobotsTxt: true,
  robotsTxtOptions: {
    policies: [{ userAgent: '*', allow: '/' }],
  },
};
