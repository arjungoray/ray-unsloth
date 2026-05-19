// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: ['intro', 'quickstart', 'architecture', 'configuration'],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/sft',
        'guides/rl',
        'guides/checkpoints',
        'guides/runtimes',
        'guides/examples',
        'guides/extending',
        'guides/testing',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/service-client',
        'api/training-client',
        'api/sampling-client',
        'api/rest-client',
        'api/types',
        'api/losses',
      ],
    },
    {
      type: 'category',
      label: 'Compatibility',
      items: ['compare-tinker', 'tech-stack'],
    },
    {
      type: 'category',
      label: 'Project',
      items: ['project/current-status', 'project/roadmap', 'project/work-in-progress'],
    },
  ],
};

module.exports = sidebars;
