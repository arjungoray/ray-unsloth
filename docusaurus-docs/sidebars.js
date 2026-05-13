// @ts-check

const sidebars = {
  tutorialSidebar: [
    'intro',
    'quickstart',
    'architecture',
    'tech-stack',
    'configuration',
    'compare-tinker',
    {
      type: 'category',
      label: 'API Primitives',
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
      label: 'Guides',
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
      label: 'Project',
      items: [
        'project/current-status',
        'project/roadmap',
        'project/work-in-progress',
      ],
    },
  ],
};

module.exports = sidebars;
