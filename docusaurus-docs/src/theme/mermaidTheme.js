/** Shared Mermaid init presets injected by src/theme/Mermaid/index.js */

export const FLOWCHART = {
  htmlLabels: true,
  curve: 'basis',
  padding: 20,
  nodeSpacing: 48,
  rankSpacing: 64,
  useMaxWidth: true,
};

export const LIGHT_THEME = {
  theme: 'base',
  look: 'classic',
  themeVariables: {
    darkMode: false,
    background: '#ffffff',
    primaryColor: '#dbeafe',
    primaryTextColor: '#1e3a8a',
    primaryBorderColor: '#2563eb',
    secondaryColor: '#d1fae5',
    secondaryTextColor: '#065f46',
    secondaryBorderColor: '#059669',
    tertiaryColor: '#ede9fe',
    tertiaryTextColor: '#5b21b6',
    tertiaryBorderColor: '#7c3aed',
    lineColor: '#64748b',
    textColor: '#0f172a',
    mainBkg: '#eff6ff',
    nodeBorder: '#2563eb',
    clusterBkg: '#f8fafc',
    clusterBorder: '#cbd5e1',
    titleColor: '#0f172a',
    edgeLabelBackground: '#ffffff',
    fontFamily:
      'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
    fontSize: '15px',
  },
  flowchart: FLOWCHART,
};

export const DARK_THEME = {
  theme: 'base',
  look: 'classic',
  themeVariables: {
    darkMode: true,
    background: '#0b1220',
    primaryColor: '#1e3a5f',
    primaryTextColor: '#bfdbfe',
    primaryBorderColor: '#3b82f6',
    secondaryColor: '#064e3b',
    secondaryTextColor: '#a7f3d0',
    secondaryBorderColor: '#10b981',
    tertiaryColor: '#4c1d95',
    tertiaryTextColor: '#ddd6fe',
    tertiaryBorderColor: '#8b5cf6',
    lineColor: '#94a3b8',
    textColor: '#e2e8f0',
    mainBkg: '#1e293b',
    nodeBorder: '#3b82f6',
    clusterBkg: '#111827',
    clusterBorder: '#334155',
    titleColor: '#f1f5f9',
    edgeLabelBackground: '#1e293b',
    fontFamily:
      'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
    fontSize: '15px',
  },
  flowchart: FLOWCHART,
};
