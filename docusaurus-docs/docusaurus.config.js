// @ts-check

const config = {
  title: 'ray-unsloth',
  tagline: 'Tinker-shaped low-level training primitives on Ray, Modal, and Unsloth',
  url: 'https://ray-unsloth.local',
  baseUrl: '/',
  organizationName: 'ray-unsloth',
  projectName: 'ray-unsloth',
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
    navbar: {
      title: 'ray-unsloth',
      items: [
        {to: '/quickstart', label: 'Quickstart', position: 'left'},
        {to: '/project/current-status', label: 'Status', position: 'left'},
        {to: '/compare-tinker', label: 'Tinker Compare', position: 'left'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Overview', to: '/'},
            {label: 'Architecture', to: '/architecture'},
            {label: 'API Reference', to: '/api/service-client'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'Configuration', to: '/configuration'},
            {label: 'Examples', to: '/guides/examples'},
            {label: 'Roadmap', to: '/project/roadmap'},
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
