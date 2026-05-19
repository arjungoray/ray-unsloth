// @ts-check

const organizationName = 'arjungoray';
const projectName = 'ray-unsloth';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'ray-unsloth',
  tagline: 'Tinker-shaped training primitives on Ray, Modal, and Unsloth',
  url: 'https://arjungoray.github.io',
  baseUrl: '/ray-unsloth/',
  organizationName,
  projectName,
  trailingSlash: false,

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: undefined,
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'ray-unsloth',
      hideOnScroll: true,
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {to: '/quickstart', label: 'Quickstart', position: 'left'},
        {to: '/compare-tinker', label: 'Tinker API', position: 'left'},
        {
          href: 'https://github.com/arjungoray/ray-unsloth',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {label: 'Overview', to: '/'},
            {label: 'Quickstart', to: '/quickstart'},
            {label: 'Architecture', to: '/architecture'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'API', to: '/api/service-client'},
            {label: 'Configuration', to: '/configuration'},
            {label: 'Tinker compatibility', to: '/compare-tinker'},
          ],
        },
        {
          title: 'Guides',
          items: [
            {label: 'SFT', to: '/guides/sft'},
            {label: 'RL', to: '/guides/rl'},
            {label: 'Examples', to: '/guides/examples'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} ray-unsloth contributors.`,
    },
    prism: {
      additionalLanguages: ['bash', 'python', 'yaml'],
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
  },
};

module.exports = config;
