import React from 'react';
import Mermaid from '@theme-original/Mermaid';
import {useColorMode} from '@docusaurus/theme-common';
import {DARK_THEME, LIGHT_THEME} from '../mermaidTheme';

function injectInit(code, config) {
  if (code.trimStart().startsWith('%%{init:')) {
    return code;
  }
  return `%%{init: ${JSON.stringify(config)}}%%\n${code}`;
}

/** @param {import('@theme/Mermaid').Props} props */
export default function MermaidWrapper(props) {
  const {colorMode} = useColorMode();
  const config = colorMode === 'dark' ? DARK_THEME : LIGHT_THEME;
  const value = injectInit(props.value, config);
  return <Mermaid {...props} value={value} />;
}
