import { Head, Html, Main, NextScript } from 'next/document';

import { withBasePath } from '@/lib/site-config';
import { siteThemeBootstrapScript } from '@/lib/site-theme';

export default function Document() {
  return (
    <Html
      lang='en-US'
      suppressHydrationWarning
      style={
        {
          '--font-inter-src': `url('${withBasePath(
            '/fonts/inter-var-latin.woff2',
          )}') format('woff2')`,
          '--newtab-cursor': `url('${withBasePath(
            '/images/new-tab.png',
          )}') 10 10, pointer`,
          '--font-display-src': `url('${withBasePath(
            '/fonts/plus-jakarta-sans-latin-700-800.woff2',
          )}') format('woff2')`,
          '--font-mono-regular-src': `url('${withBasePath(
            '/fonts/ibm-plex-mono-latin-400.woff2',
          )}') format('woff2')`,
          '--font-mono-semibold-src': `url('${withBasePath(
            '/fonts/ibm-plex-mono-latin-600.woff2',
          )}') format('woff2')`,
        } as Record<string, string>
      }
    >
      <Head>
        <link
          rel='preload'
          href={withBasePath('/fonts/inter-var-latin.woff2')}
          as='font'
          type='font/woff2'
          crossOrigin='anonymous'
        />
        <link
          rel='preload'
          href={withBasePath('/fonts/plus-jakarta-sans-latin-700-800.woff2')}
          as='font'
          type='font/woff2'
          crossOrigin='anonymous'
        />
      </Head>
      <body>
        <script
          dangerouslySetInnerHTML={{ __html: siteThemeBootstrapScript() }}
        />
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
