import { Head, Html, Main, NextScript } from 'next/document';

import { withBasePath } from '@/lib/site-config';

export default function Document() {
  return (
    <Html
      lang='en'
      style={
        {
          '--font-inter-src': `url('${withBasePath(
            '/fonts/inter-var-latin.woff2',
          )}') format('woff2')`,
          '--newtab-cursor': `url('${withBasePath(
            '/images/new-tab.png',
          )}') 10 10, pointer`,
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
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
