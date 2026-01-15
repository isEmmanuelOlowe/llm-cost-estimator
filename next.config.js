/** @type {import('next').NextConfig} */
const isProd = process.env.NODE_ENV === 'production';
const baseName = process.env.NEXT_PUBLIC_BASE_PATH || '';
const basePath = isProd ? baseName : '';
const assetPrefix = isProd && baseName ? `${baseName}/` : undefined;

const nextConfig = {
  eslint: {
    dirs: ['src'],
  },

  reactStrictMode: true,
  swcMinify: true,

  output: 'export',
  basePath,
  assetPrefix,
  trailingSlash: true,

  images: {
    unoptimized: true,
  },

  // SVGR
  webpack(config) {
    config.module.rules.push({
      test: /\.svg$/i,
      issuer: /\.[jt]sx?$/,
      use: [
        {
          loader: '@svgr/webpack',
          options: {
            typescript: true,
            icon: true,
          },
        },
      ],
    });

    return config;
  },
};

module.exports = nextConfig;
