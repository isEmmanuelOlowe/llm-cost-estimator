import Head from 'next/head';
import { useRouter } from 'next/router';

import {
  absoluteUrl,
  resolveAssetUrl,
  siteDescription,
  siteName,
  withBasePath,
} from '@/lib/site-config';

const defaultMeta = {
  title: siteName,
  siteName,
  description: siteDescription,
  url: absoluteUrl('/'),
  type: 'website',
  robots: 'follow, index',
  image: resolveAssetUrl('/images/large-og.png'),
};

type SeoProps = {
  date?: string;
  templateTitle?: string;
} & Partial<typeof defaultMeta>;

export default function Seo(props: SeoProps) {
  const router = useRouter();
  const routePath = router.asPath.split('?')[0] || '/';
  const pageUrl = absoluteUrl(routePath);
  const meta = {
    ...defaultMeta,
    ...props,
  };
  meta['title'] = props.templateTitle
    ? `${props.templateTitle} | ${meta.siteName}`
    : meta.title;
  meta['url'] = pageUrl;
  meta['image'] = resolveAssetUrl(meta.image);

  return (
    <Head>
      <title>{meta.title}</title>
      <meta name='robots' content={meta.robots} />
      <meta content={meta.description} name='description' />
      <meta property='og:url' content={meta.url} />
      <link rel='canonical' href={meta.url} />
      {/* Open Graph */}
      <meta property='og:type' content={meta.type} />
      <meta property='og:site_name' content={meta.siteName} />
      <meta property='og:description' content={meta.description} />
      <meta property='og:title' content={meta.title} />
      <meta name='image' property='og:image' content={meta.image} />
      {/* Twitter */}
      <meta name='twitter:card' content='summary_large_image' />
      {/* <meta name='twitter:site' content='@phophet_ai' /> */}
      <meta name='twitter:title' content={meta.title} />
      <meta name='twitter:description' content={meta.description} />
      <meta name='twitter:image' content={meta.image} />
      {meta.date && (
        <>
          <meta property='article:published_time' content={meta.date} />
          <meta
            name='publish_date'
            property='og:publish_date'
            content={meta.date}
          />
          <meta
            name='author'
            property='article:author'
            content='Emmanuel Olowe'
          />
        </>
      )}

      {/* Favicons */}
      {favicons.map((linkProps) => (
        <link key={linkProps.href} {...linkProps} />
      ))}
      <meta name='msapplication-TileColor' content='#ffffff' />
      <meta
        name='msapplication-config'
        content={withBasePath('/favicon/browserconfig.xml')}
      />
      <meta name='theme-color' content='#ffffff' />
    </Head>
  );
}

const favicons: Array<React.ComponentPropsWithoutRef<'link'>> = [
  {
    rel: 'apple-touch-icon',
    sizes: '180x180',
    href: withBasePath('/favicon/apple-touch-icon.png'),
  },
  {
    rel: 'icon',
    type: 'image/png',
    sizes: '32x32',
    href: withBasePath('/favicon/favicon-32x32.png'),
  },
  {
    rel: 'icon',
    type: 'image/png',
    sizes: '16x16',
    href: withBasePath('/favicon/favicon-16x16.png'),
  },
  { rel: 'manifest', href: withBasePath('/favicon/site.webmanifest') },
  { rel: 'shortcut icon', href: withBasePath('/favicon/favicon.ico') },
];
